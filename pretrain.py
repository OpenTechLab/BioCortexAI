# pretrain.py
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import sentencepiece as spm
import config
from model import Transformer

# ---------------- Utils ----------------

def ddp_setup():
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend="nccl")

def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def generate_sample(model, sp_tokenizer, device, prompt, max_new_tokens=50):
    model.eval()
    ids = [sp_tokenizer.bos_id()] + sp_tokenizer.encode_as_ids(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        x_cond = x[:, -config.pretrain_max_seq_len:] if x.shape[1] > config.pretrain_max_seq_len else x
        logits = model(x_cond, pad_id=sp_tokenizer.pad_id(), prompt_text=prompt)
        last = logits[:, -1, :]
        probs = torch.softmax(last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        if next_id.item() == sp_tokenizer.eos_id():
            break
        x = torch.cat([x, next_id], dim=1)
    out = sp_tokenizer.decode(x[0].tolist())
    model.train()
    return out

@torch.no_grad()
def validate(model, val_dataloader, criterion, device, sp_tokenizer):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(val_dataloader, desc="Validation", leave=False)
    for batch in pbar:
        batch = batch.to(device, non_blocking=True)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        logits = model(inputs, pad_id=sp_tokenizer.pad_id()) 
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / max(1, len(val_dataloader))

class ChunkedDataset(Dataset):
    def __init__(self, bin_file_path, chunk_size):
        flat_data = np.memmap(bin_file_path, dtype=np.uint16, mode='r')
        num_complete_chunks = len(flat_data) // chunk_size
        num_tokens_to_keep = num_complete_chunks * chunk_size
        self.data = flat_data[:num_tokens_to_keep].reshape(-1, chunk_size)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].astype(np.int64))

def save_checkpoint(path, model, optimizer, epoch, iter_num, loss, is_ddp):
    if int(os.environ.get("RANK", 0)) != 0:
        return
    state = model.module.state_dict() if is_ddp else model.state_dict()
    torch.save({
        "epoch": epoch,
        "iter_num": iter_num,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"Checkpoint uložen do {path}")

def load_checkpoint(checkpoint_path, model, optimizer, is_ddp):
    if not os.path.exists(checkpoint_path):
        return 0, 0
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    unwanted_prefix = "_orig_mod."
    for k in list(state.keys()):
        if k.startswith(unwanted_prefix):
            state[k[len(unwanted_prefix):]] = state.pop(k)
    if is_ddp:
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    iter_num = ckpt.get("iter_num", 0)
    print(f"Checkpoint loaded {checkpoint_path}. Continuing from epoch {start_epoch} and step {iter_num}.")
    return start_epoch, iter_num

# ---------------- Main ----------------

def main():
    is_ddp = 'WORLD_SIZE' in os.environ
    ddp_setup()
    if is_ddp:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        device = f"cuda:{local_rank}"
    else:
        world_size, rank = 1, 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if rank == 0:
        print(f"Runs on device: {device}. Number of GPUs: {world_size}")

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(f"{config.model_prefix}.model")
    actual_vocab_size = sp.get_piece_size()
    if rank == 0:
        print(f"Real vocab size: {actual_vocab_size}")

    # Data
    train_bin_path = config.chunked_binary_path.replace(".bin", "_train.bin")
    val_bin_path = config.chunked_binary_path.replace(".bin", "_val.bin")
    if not (os.path.exists(train_bin_path) and os.path.exists(val_bin_path)):
        if rank == 0:
            print(f"Chyba: Tréninkový ({train_bin_path}) nebo validační ({val_bin_path}) soubor neexistuje.")
            print("Nejprve spusťte skripty z 'nastroje_pro_data/'.")
        ddp_cleanup()
        return

    train_dataset = ChunkedDataset(train_bin_path, config.pretrain_max_seq_len)
    val_dataset = ChunkedDataset(val_bin_path, config.pretrain_max_seq_len)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.pretrain_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.pretrain_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    steps_per_epoch = len(train_loader)
    lr_decay_iters = config.pretrain_num_epochs * steps_per_epoch
    if rank == 0:
        print(f"Total number of training steps: {lr_decay_iters}")

    # Model
    original_use_plant_network = config.USE_PLANT_NETWORK 
    config.USE_PLANT_NETWORK = False 


    model = Transformer(
        vocab_size=actual_vocab_size,
        dim=config.embedding_dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        ff_multiplier=config.ff_dim_multiplier,
        dropout=config.dropout,
    ).to(device)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model size: {num_params:,} parameters")


    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.pretrain_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=getattr(config, "weight_decay", 0.0),
    )
    criterion = nn.CrossEntropyLoss()

    # Checkpoint
    start_epoch, iter_num = 0, 0
    best_val_loss = float('inf')
    ckpt_dir = config.pretrain_checkpoint_dir

    if rank == 0 and getattr(config, "resume_from_checkpoint", False):
        latest = os.path.join(ckpt_dir, "latest_checkpoint.pt")
        if os.path.exists(latest):
            start_epoch, iter_num = load_checkpoint(latest, model, optimizer, is_ddp)
        bestp = os.path.join(ckpt_dir, "best_checkpoint.pt")
        if os.path.exists(bestp):
            best_val_loss = torch.load(bestp, map_location="cpu")["loss"]
            print(f"Best validation loss: {best_val_loss:.3f}")

    model.train()
    for epoch in range(start_epoch, config.pretrain_num_epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        it = tqdm(train_loader, desc=f"Pretrain epoch {epoch+1}/{config.pretrain_num_epochs}") if rank == 0 else train_loader
        for batch in it:
            batch = batch.to(device, non_blocking=True)
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()

            lr = get_lr(iter_num, config.pretrain_warmup_iters, config.pretrain_learning_rate, lr_decay_iters, config.pretrain_min_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, pad_id=sp.pad_id())  
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            iter_num += 1
            epoch_loss += loss.item()

        if rank == 0:
            avg_train = epoch_loss / max(1, len(train_loader))
            model_to_val = model.module if is_ddp else model
            avg_val = validate(model_to_val, val_loader, criterion, device, sp)

            print("-" * 70)
            print(f"EPOCH {epoch+1:02} FINISHED | Training loss: {avg_train:.3f} | Validation loss: {avg_val:.3f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                print(f"New best validation loss: {best_val_loss:.3f}. Saving 'best_checkpoint.pt'.")
                save_checkpoint(os.path.join(ckpt_dir, "best_checkpoint.pt"), model, optimizer, epoch + 1, iter_num, best_val_loss, is_ddp)

            print("--- Generating sample after pretrain ---")
            prompt = "Byl jednou jeden"
            sample = generate_sample(model_to_val, sp, device, prompt)
            print(f"Prompt: '{prompt}'\nVygenerováno: {sample}")
            print("-" * 70)

            save_checkpoint(os.path.join(ckpt_dir, "latest_checkpoint.pt"), model, optimizer, epoch + 1, iter_num, avg_train, is_ddp)

    ddp_cleanup()

if __name__ == "__main__":
    main()
