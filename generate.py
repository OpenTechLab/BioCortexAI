# generate.py
import os
import torch
import sentencepiece as spm
import config
from model import Transformer


@torch.no_grad()
def generate(model, sp_tokenizer, device, input_token_ids, new_input_tokens=None, max_new_tokens=100, temperature=0.8):

    model.eval()

    context_tokens = input_token_ids[-(config.pretrain_max_seq_len - max_new_tokens):]
    initial_len = len(context_tokens)
    input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
    last_sentiment = None
    
    if new_input_tokens:
        subrsting_for_sentiment = context_tokens[-100:] if len(context_tokens) > 100 else context_tokens
        prompt_for_sentiment = sp_tokenizer.decode(subrsting_for_sentiment)
        #print(f"\nDo sentimentu jde ({prompt_for_sentiment})\n")
    else:
        prompt_for_sentiment = ""

    for _ in range(max_new_tokens):

        input_cond = input_ids

        logits, hidden_states = model(input_cond, return_hidden=True)

        if getattr(model, "use_plant_net", False) and getattr(model, "plant_layer", None):
            last_sentiment = model.plant_layer.update_state(logits, hidden_states, prompt_for_sentiment)

        last_token_logits = logits[:, -1, :]

        if temperature and temperature > 0:
            probs = torch.softmax(last_token_logits / float(temperature), dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(last_token_logits, dim=-1, keepdim=True)

        if next_token_id.item() == sp_tokenizer.eos_id():
            break
            
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    all_final_tokens = input_ids[0].tolist()
    
    newly_generated_tokens = all_final_tokens[initial_len:]
    
    response = sp_tokenizer.decode(newly_generated_tokens)

    hormones = model.plant_layer.get_global_hormones() if getattr(model, "plant_layer", None) else None
    
    return response, hormones, last_sentiment, all_final_tokens


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Used device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(f"{config.model_prefix}.model")
    actual_vocab_size = sp.get_piece_size()

    model = Transformer(
        vocab_size=actual_vocab_size,
        dim=config.embedding_dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        ff_multiplier=config.ff_dim_multiplier,
        dropout=config.dropout,
    ).to(device)

    checkpoint_path = config.generator_checkpoint_path
    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    else:
        print(f"ERROR: Checkpoint '{checkpoint_path}' was not found.")
        return

    if getattr(model, "use_plant_net", False):
        print("\nThe model with an active plant network is ready.")
    else:
        print("\nModel runs in standard mode (Without the plant network.).")

    system_prompt = "Tvé jméno je Ája. Jsi prototyp BioCortexAI."
    prompt_formatted = f"system: {system_prompt} "
    conversation_tokens = [sp.bos_id()] + sp.encode_as_ids(prompt_formatted)

    print("\nInput text (or 'exit' for end).")
    try:
        while True:
            prompt = input("User: ")
            if prompt.lower() in ["konec", "exit", "quit"]:
                break

            prompt_formatted = f"user: {prompt} model:"
            new_tokens = sp.encode_as_ids(prompt_formatted)
            
            input_for_generate = conversation_tokens + new_tokens

            response, hormones, sentiment, updated_tokens = generate(model, sp, device, input_for_generate, new_input_tokens=new_tokens)
            
            conversation_tokens = updated_tokens
            
            print(f"Model: {response}")
            if hormones:
                hormone_str = " | ".join([f"{k.capitalize()}: {v:.2f}" for k, v in hormones.items()])
                sentiment_str = f"Sentiment: {sentiment:.2f}" if sentiment is not None else "Sentiment: N/A"
                print(f"[MODEL MOOD] {hormone_str} | {sentiment_str}")
    finally:
        if getattr(model, "plant_layer", None):
            model.plant_layer.save_state()


if __name__ == "__main__":
    main()