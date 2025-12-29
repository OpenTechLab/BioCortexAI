# finetune.py
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import sentencepiece as spm
import config
from model import Transformer

def ddp_setup():
    dist.init_process_group(backend="nccl")

def ddp_cleanup():
    dist.destroy_process_group()

def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):

    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def save_checkpoint(checkpoint_dir, model, optimizer, epoch, last_loss):

    if int(os.environ.get("RANK", 0)) != 0:
        return  
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": last_loss,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved into {checkpoint_path}")

def load_finetune_checkpoint(checkpoint_path, model, optimizer):

    if not os.path.exists(checkpoint_path):
        return 0, 0.0
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["model_state_dict"]
    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    last_loss = checkpoint["loss"]
    print(f"Continuing fine-tuning from {checkpoint_path} from epoch {start_epoch}.")
    return start_epoch, last_loss

@torch.no_grad()
def generate_sample(model, sp_tokenizer, device, prompt, max_new_tokens=50):

    model.eval()
    prompt_formatted = f"user: {prompt} model:"
    prompt_tokens = [sp_tokenizer.bos_id()] + sp_tokenizer.encode_as_ids(prompt_formatted)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        input_cond = input_ids if input_ids.size(1) <= config.pretrain_max_seq_len else input_ids[:, -config.pretrain_max_seq_len :]

        logits = model(input_cond, pad_id=sp_tokenizer.eos_id())
        last_token_logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        if next_token_id.item() == sp_tokenizer.eos_id():
            break

        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    output_text = sp_tokenizer.decode(input_ids[0].tolist())
    model.train()
    return output_text


class FineTuneDataset(Dataset):

    def __init__(self, filepath, sp_tokenizer, max_len):
        self.sp_tokenizer = sp_tokenizer
        self.max_len = max_len
        self.dialogues = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if "user:" in line and "model:" in line:
                    parts = line.split("model:")
                    user_part = "user: " + parts[0].replace("user:", "").strip()
                    model_part = " model: " + parts[1].strip()

                    user_tokens = self.sp_tokenizer.encode_as_ids(user_part)
                    model_tokens = self.sp_tokenizer.encode_as_ids(model_part)

                    full_tokens = [self.sp_tokenizer.bos_id()] + user_tokens + model_tokens + [self.sp_tokenizer.eos_id()]

                    if len(full_tokens) > self.max_len:
                        continue

                    self.dialogues.append(
                        {
                            "tokens": torch.tensor(full_tokens, dtype=torch.long),
                            "user_len": len(user_tokens) + 1,  # +1 for BOS token
                        }
                    )

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        return self.dialogues[idx]

class FineTunePadCollator:

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(item["tokens"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)  

        for i, item in enumerate(batch):
            tokens = item["tokens"]
            seq_len, user_len = len(tokens), item["user_len"]
            input_ids[i, :seq_len] = tokens

            labels[i, user_len:seq_len] = tokens[user_len:seq_len]

        return {"input_ids": input_ids, "labels": labels}

def main():
    is_ddp = "WORLD_SIZE" in os.environ
    if is_ddp:
        ddp_setup()
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        device = f"cuda:{local_rank}"
    else:
        world_size = 1
        rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        print(f"Initiating fine-tuning on device: {device}. Number of GPUs: {world_size}")

    sp = spm.SentencePieceProcessor()
    sp.load(f"{config.model_prefix}.model")
    actual_vocab_size = sp.get_piece_size()

    dataset = FineTuneDataset(config.finetune_data_path, sp, config.pretrain_max_seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None


    collator = FineTunePadCollator(pad_id=sp.eos_id())

    dataloader = DataLoader(
        dataset,
        batch_size=config.finetune_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collator,
    )


    num_iterations_per_epoch = len(dataloader)
    lr_decay_iters = config.finetune_num_epochs * num_iterations_per_epoch
    if rank == 0:
        print(f"Celkový počet fine-tuning kroků: {lr_decay_iters}")
        print(f"Warmup bude trvat {config.finetune_warmup_iters} kroků.")

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
    )


    if os.path.exists(config.base_model_path):
        if rank == 0:
            print(f"Loading weights from the base model: {config.base_model_path}")
        checkpoint = torch.load(config.base_model_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict, strict=True)
    else:
        if rank == 0:
            print(f"Warning: Base model '{config.base_model_path}' was not found. I am training from scratch (random initialization).")

    model.to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.AdamW(model.parameters(), lr=config.finetune_learning_rate)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())  # ready for AMP (not used yet)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    start_epoch = 0
    iter_num = 0
    if rank == 0 and config.resume_from_checkpoint:
        checkpoint_path = os.path.join(config.finetune_checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            start_epoch, _ = load_finetune_checkpoint(checkpoint_path, model, optimizer)

    if is_ddp:
        start_epoch_tensor = torch.tensor(start_epoch, device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()

    model.train()
    for epoch in range(start_epoch, config.finetune_num_epochs):
        epoch_loss = 0
        if is_ddp:
            sampler.set_epoch(epoch)

        pbar = dataloader
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Fine-tuning Epoch {epoch+1}/{config.finetune_num_epochs}")

        for i, batch in enumerate(pbar):
            lr = get_lr(
                iter_num,
                config.finetune_warmup_iters,
                config.finetune_learning_rate,
                lr_decay_iters,
                config.finetune_min_lr,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            input_seq = inputs[:, :-1].contiguous()
            target_seq = labels[:, 1:].contiguous()

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_seq, pad_id=sp.eos_id())
            loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            iter_num += 1
            epoch_loss += loss.item()

        if rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch: {epoch+1:02} | Average loss: {avg_loss:.3f} | Last LR: {lr:.7f}")

            print("--- Generating a sample after fine-tuning. ---")
            prompt = "Existují chytré triky na počítání?"

            generated_text = generate_sample(model.module if is_ddp else model, sp, device, prompt)
            print(f"Prompt: '{prompt}'")
            print(f"Generated: {generated_text}")
            print("-" * 50)

            save_checkpoint(config.finetune_checkpoint_dir, model, optimizer, epoch, avg_loss)

    if is_ddp:
        ddp_cleanup()


if __name__ == "__main__":
    main()
