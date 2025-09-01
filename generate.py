import os
import torch
import sentencepiece as spm
import config
from model import Transformer
from sentiment_analyzer import sentiment_analyzer

def get_mood_visualization(hormones):
    """
    Vstup: dict {'dopamin','serotonin','kortizol','oxytocin'} s typ. rozsahem 0.5–1.5 (neutral=1.0).
    Výstup: (emoji, hex barva). Logika respektuje vliv hormonů v modelu:
      - dopamin: škáluje V-vektory (arousal/drive)  → Attention V *= dopamin
      - serotonin, oxytocin: zesilují rezidua (pozitivní valence / sociální otevřenost)
      - kortizol: brzdí rezidua (stres/napětí)
    """
    if not hormones:
        return "😐", "#F5F5F5"

    # Bezpečné načtení + normalizace do ⟨-1,1⟩ vzhledem k neutral=1.0 a rozsahu 0.5–1.5
    d = {
        'dopamin': float(hormones.get('dopamin', 1.0)),
        'serotonin': float(hormones.get('serotonin', 1.0)),
        'kortizol': float(hormones.get('kortizol', 1.0)),
        'oxytocin': float(hormones.get('oxytocin', 1.0)),
    }
    n = {k: max(-1.0, min(1.0, (v - 1.0) / 0.5)) for k, v in d.items()}  # -1..1

    # Jemnější prahy (EMA + homeostáza drží hodnoty blízko 1.0)
    TM, TS = 0.30, 0.60  # mírná / silná odchylka

    # --- Kombinované stavy (priorita před single) ---
    # „Radostné spojení“: vysoký serotonin & oxytocin, žádný zvýšený stres
    if n['serotonin'] >= TM and n['oxytocin'] >= TM and n['kortizol'] <= 0.0:
        return "🥳", "#EAF8E6"

    # „Napjaté vzrušení“: dopamin + kortizol výš
    if n['dopamin'] >= TM and n['kortizol'] >= TM:
        return "😲", "#FFE3C3"

    # „Stres & útlum“: kortizol výš, serotonin níž
    if n['kortizol'] >= TM and n['serotonin'] <= -TM:
        return "😣", "#FFE5E5"

    # „Neosobní hnací stav“: dopamin výš, oxytocin níž
    if n['dopamin'] >= TM and n['oxytocin'] <= -TM:
        return "😶‍🌫️", "#FFF1E0"

    # --- Projekce na (valence, arousal) ---
    # Valence ~ serotonin + (0.6)*oxytocin − (0.8)*↑kortizol + (0.3)*↓kortizol
    valence = (
        n['serotonin']
        + 0.6 * n['oxytocin']
        - 0.8 * max(n['kortizol'], 0.0)
        + 0.3 * min(n['kortizol'], 0.0)
    )
    # Arousal ~ dopamin + 0.5*↑kortizol (stres zvyšuje napětí)
    arousal = n['dopamin'] + 0.5 * max(n['kortizol'], 0.0)

    if valence >= TS and arousal >= TM:
        return "🤩", "#FFFDE7"   # silně pozitivní, lehce vzrušený
    if valence >= TM:
        return "😊", "#F1F8E9"   # klidně pozitivní
    if valence <= -TS and arousal >= TM:
        return "😰", "#E0F3FF"   # úzkost
    if valence <= -TM:
        return "😔", "#FFF8DC"   # smutek/stažení

    # --- Jednoduché „dominantní“ stavy (mírné odchylky) ---
    dominant = max(n.items(), key=lambda kv: abs(kv[1]))
    h, s = dominant
    if h == 'dopamin':
        return ("🤗", "#FFE4E1") if s > 0 else ("😴", "#D3D3D3")
    if h == 'serotonin':
        return ("😊", "#98FB98") if s > 0 else ("😔", "#F0E68C")
    if h == 'kortizol':
        return ("😰", "#E0FFFF") if s > 0 else ("😌", "#F0F8FF")
    if h == 'oxytocin':
        return ("🥰", "#DDA0DD") if s > 0 else ("😕", "#FFF8DC")

    # Fallback
    return "😐", "#F5F5F5"

@torch.no_grad()
def generate(model, sp_tokenizer, device, input_token_ids, new_input_tokens=None, max_new_tokens=100, temperature=0.8):
    model.eval()
    
    # Sentiment analýza jen když přijde nový input
    should_analyze_sentiment = bool(new_input_tokens)

    context_tokens = input_token_ids[-(config.pretrain_max_seq_len - max_new_tokens):]
    initial_len = len(context_tokens)
    input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
    last_sentiment = None
    
    if new_input_tokens:
        subrsting_for_sentiment = context_tokens[-100:] if len(context_tokens) > 100 else context_tokens
        prompt_for_sentiment = sp_tokenizer.decode(subrsting_for_sentiment)
    else:
        prompt_for_sentiment = ""

    for _ in range(max_new_tokens):
        input_cond = input_ids
        logits, hidden_states = model(input_cond, return_hidden=True)

        if getattr(model, "use_plant_net", False) and getattr(model, "plant_layer", None):
            model.plant_layer.update_state(
                logits, hidden_states,
                prompt_for_sentiment if should_analyze_sentiment else ""
            )
        # 2) Lazy sentiment – zavolat přesně 1× na nový uživatelský vstup
        if should_analyze_sentiment:
            last_sentiment = sentiment_analyzer.get_score(prompt_for_sentiment)
            should_analyze_sentiment = False

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
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    else:
        print(f"ERROR: Checkpoint '{checkpoint_path}' was not found.")
        return

    if getattr(model, "use_plant_net", False):
        print("\nThe model with an active plant network is ready.")
    else:
        print("\nModel runs in standard mode (Without the plant network).")

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
                mood_emoji, mood_color = get_mood_visualization(hormones)
                hormone_str = " | ".join([f"{k.capitalize()}: {v:.2f}" for k, v in hormones.items()])
                sentiment_str = f"Sentiment: {sentiment:.2f}" if sentiment is not None else "Sentiment: N/A"
                print(f"[MODEL MOOD] {mood_emoji} | {hormone_str} | {sentiment_str}")
    finally:
        if getattr(model, "plant_layer", None):
            model.plant_layer.save_state()

if __name__ == "__main__":
    main()
