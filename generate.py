# generate.py
# -*- coding: utf-8 -*-
"""
BioCortexAI Generator with Mirror Integration
==============================================
CLI generation script with predictive mirror loop.

(c) 2025 OpenTechLab Jablonec nad Nisou s.r.o.
Author: Michal Seidl
"""

import os
import torch
import sentencepiece as spm
import config
from model import Transformer
from sentiment_analyzer import sentiment_analyzer

# Import Mirror Integration
try:
    from mirror_integration import (
        mirror_integration,
        prepare_mirror_context,
        MirrorIntegration,
    )
    MIRROR_AVAILABLE = True
except ImportError:
    MIRROR_AVAILABLE = False
    print("Warning: Mirror integration not available")


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
    # „Radostné spojení": vysoký serotonin & oxytocin, žádný zvýšený stres
    if n['serotonin'] >= TM and n['oxytocin'] >= TM and n['kortizol'] <= 0.0:
        return "🥳", "#EAF8E6"

    # „Napjaté vzrušení": dopamin + kortizol výš
    if n['dopamin'] >= TM and n['kortizol'] >= TM:
        return "😲", "#FFE3C3"

    # „Stres & útlum": kortizol výš, serotonin níž
    if n['kortizol'] >= TM and n['serotonin'] <= -TM:
        return "😣", "#FFE5E5"

    # „Neosobní hnací stav": dopamin výš, oxytocin níž
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

    # --- Jednoduché „dominantní" stavy (mírné odchylky) ---
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
def generate_base(model, sp_tokenizer, device, input_token_ids, max_new_tokens=100, temperature=0.8, return_all_hidden=False):
    """
    Basic generation function - generates tokens and optionally returns all hidden states.
    
    Args:
        model: Transformer model
        sp_tokenizer: SentencePiece tokenizer
        device: torch device
        input_token_ids: Input token IDs
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        return_all_hidden: If True, collect hidden states for all generated tokens
    
    Returns:
        Tuple of (response_text, hormones, all_tokens, collected_hidden_states)
        collected_hidden_states is None if return_all_hidden=False
    """
    model.eval()
    
    context_tokens = input_token_ids[-(config.pretrain_max_seq_len - max_new_tokens):]
    initial_len = len(context_tokens)
    input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
    
    collected_hidden = [] if return_all_hidden else None
    
    for _ in range(max_new_tokens):
        logits, hidden_states = model(input_ids, return_hidden=True)
        
        # Collect hidden states of the newly generated position
        if return_all_hidden:
            collected_hidden.append(hidden_states[:, -1:, :].clone())
        
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
    
    # Stack collected hidden states if needed
    stacked_hidden = None
    if return_all_hidden and collected_hidden:
        stacked_hidden = torch.cat(collected_hidden, dim=1)  # [1, seq_len, hidden_dim]
    
    return response, hormones, all_final_tokens, stacked_hidden


@torch.no_grad()
def generate_with_mirror(
    model, 
    sp_tokenizer, 
    device, 
    conversation_tokens,
    new_input_tokens,
    previous_user_hidden=None,
    max_new_tokens=100, 
    temperature=0.8
):
    """
    Generation with Mirror Prediction Loop.
    
    Flow:
    1. Generate response (hidden from user)
    2. Mirror-swap the response + context
    3. Generate expected user response
    4. Store expectation vectors
    5. Return original response for user display
    6. (Next turn) Compare actual user response with expectation
    
    Args:
        model: Transformer model
        sp_tokenizer: SentencePiece tokenizer
        device: torch device
        conversation_tokens: Full conversation token history
        new_input_tokens: New user input tokens (for this turn)
        previous_user_hidden: Hidden states from processing previous user input 
                              (for comparison with expectation)
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Dict with:
            - response: Generated response text
            - hormones: Current hormone state
            - sentiment: Sentiment score
            - updated_tokens: Updated conversation tokens
            - prediction_result: Result of comparing with previous expectation (or None)
    """
    result = {
        "response": "",
        "hormones": None,
        "sentiment": None,
        "updated_tokens": conversation_tokens.copy(),
        "prediction_result": None,
        "mirror_stats": None,
    }
    
    # --- STEP 0: Compare previous expectation with actual user input ---
    if previous_user_hidden is not None and mirror_integration.expectation_buffer.is_valid():
        actual_text = sp_tokenizer.decode(new_input_tokens) if new_input_tokens else ""
        
        prediction_result = mirror_integration.compare_with_actual(
            previous_user_hidden,
            actual_text
        )
        result["prediction_result"] = prediction_result
        
        # === DEBUG OUTPUT - Comparison ===
        if getattr(config, 'MIRROR_DEBUG', False) and prediction_result:
            sep = getattr(config, 'MIRROR_DEBUG_SEPARATOR', '=' * 60)
            prefix = getattr(config, 'MIRROR_DEBUG_PREFIX', '[🪞 MIRROR DEBUG]')
            quality_emoji = {"good": "✅", "neutral": "➖", "bad": "❌"}.get(prediction_result.quality, "❓")
            print(f"\n{sep}")
            print(f"{prefix} PREDICTION COMPARISON RESULT")
            print(f"{sep}")
            print(f"{prefix} What model expected user would say:")
            print(f"    \"{mirror_integration.expectation_buffer.expected_text[:200]}...\"" if len(mirror_integration.expectation_buffer.expected_text) > 200 else f"    \"{mirror_integration.expectation_buffer.expected_text}\"")
            print(f"{prefix} What user actually said:")
            print(f"    \"{actual_text[:200]}...\"" if len(actual_text) > 200 else f"    \"{actual_text}\"")
            print(f"{sep}")
            print(f"{prefix} Comparison metrics:")
            print(f"    Prediction Error: {prediction_result.prediction_error:.4f}")
            print(f"    Cosine Similarity: {prediction_result.cosine_similarity:.4f}")
            print(f"    Quality: {quality_emoji} {prediction_result.quality.upper()}")
            print(f"{prefix} Hormone deltas applied:")
            for hormone, delta in prediction_result.hormone_deltas.items():
                sign = "+" if delta >= 0 else ""
                print(f"    {hormone}: {sign}{delta:.4f}")
            print(f"{sep}\n")
        
        # Apply to PlantNet
        if prediction_result and getattr(model, "plant_layer", None):
            mirror_integration.apply_to_plant_net(model.plant_layer, prediction_result)
    
    # Clear old expectation
    mirror_integration.clear_expectation()
    
    # --- STEP 1: Generate response (hidden from user) ---
    input_for_generate = conversation_tokens + new_input_tokens
    
    response, hormones, updated_tokens, response_hidden = generate_base(
        model, sp_tokenizer, device,
        input_for_generate,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_all_hidden=True
    )
    
    result["response"] = response
    result["updated_tokens"] = updated_tokens
    
    # Update PlantNet with the generation
    if getattr(model, "use_plant_net", False) and getattr(model, "plant_layer", None):
        # Get hidden states for PlantNet update
        logits, hidden_states = model(
            torch.tensor([updated_tokens[-(config.pretrain_max_seq_len):]], device=device),
            return_hidden=True
        )
        prompt_text = sp_tokenizer.decode(new_input_tokens) if new_input_tokens else ""
        model.plant_layer.update_state(logits, hidden_states, prompt_text)
        result["sentiment"] = sentiment_analyzer.get_score(prompt_text)
    
    result["hormones"] = model.plant_layer.get_global_hormones() if getattr(model, "plant_layer", None) else None
    
    # --- STEP 2-4: Mirror prediction loop ---
    if config.USE_MIRROR_MODULE and MIRROR_AVAILABLE:
        try:
            # Prepare mirrored context (swap deixis in context + response)
            swapped_context_text, swapped_tokens = prepare_mirror_context(
                conversation_tokens,
                response,
                sp_tokenizer,
                config.MIRROR_CONTEXT_LAST_TOKENS
            )
            
            # === DEBUG OUTPUT ===
            if getattr(config, 'MIRROR_DEBUG', False):
                sep = getattr(config, 'MIRROR_DEBUG_SEPARATOR', '=' * 60)
                prefix = getattr(config, 'MIRROR_DEBUG_PREFIX', '[🪞 MIRROR DEBUG]')
                print(f"\n{sep}")
                print(f"{prefix} MIRROR PREDICTION LOOP")
                print(f"{sep}")
                print(f"{prefix} Lambda values:")
                print(f"    λ_deixis = {config.MIRROR_LAMBDA_DEIXIS}")
                print(f"    λ_styl   = {config.MIRROR_LAMBDA_STYL}")
                print(f"{prefix} Context tokens used: {config.MIRROR_CONTEXT_LAST_TOKENS}")
                print(f"{sep}")
                print(f"{prefix} ORIGINAL MODEL RESPONSE (before showing to user):")
                print(f"    \"{response}\"")
                print(f"{sep}")
                print(f"{prefix} SWAPPED CONTEXT (after deictic swap):")
                # Show first 500 chars of swapped context
                swapped_display = swapped_context_text[:500] + "..." if len(swapped_context_text) > 500 else swapped_context_text
                for line in swapped_display.split('\n'):
                    print(f"    {line}")
                print(f"{sep}")
            
            if config.MIRROR_VERBOSE:
                print(f"[Mirror] Swapped context: {swapped_context_text[:100]}...")
            
            # Generate expected user response (what model thinks user will say)
            expected_response, _, _, expected_hidden = generate_base(
                model, sp_tokenizer, device,
                swapped_tokens,
                max_new_tokens=config.MIRROR_EXPECTATION_MAX_TOKENS,
                temperature=temperature,
                return_all_hidden=True
            )
            
            # === DEBUG OUTPUT - Expected response ===
            if getattr(config, 'MIRROR_DEBUG', False):
                sep = getattr(config, 'MIRROR_DEBUG_SEPARATOR', '=' * 60)
                prefix = getattr(config, 'MIRROR_DEBUG_PREFIX', '[🪞 MIRROR DEBUG]')
                print(f"{prefix} EXPECTED USER RESPONSE (model's prediction):")
                print(f"    \"{expected_response}\"")
                print(f"{prefix} Expected response tokens: {expected_hidden.shape[1] if expected_hidden is not None else 0}")
                print(f"{sep}\n")
            
            if config.MIRROR_VERBOSE:
                print(f"[Mirror] Expected response: {expected_response[:100]}...")
            
            # Store expectation for next turn
            if expected_hidden is not None:
                mirror_integration.store_expectation(
                    response_text=response,
                    expected_text=expected_response,
                    expected_hidden_states=expected_hidden,
                    swapped_context=swapped_context_text
                )
            
            result["mirror_stats"] = mirror_integration.get_stats()
            
        except Exception as e:
            if config.MIRROR_VERBOSE or getattr(config, 'MIRROR_DEBUG', False):
                print(f"[Mirror] Error in prediction loop: {e}")
    
    return result


@torch.no_grad()
def generate(model, sp_tokenizer, device, input_token_ids, new_input_tokens=None, max_new_tokens=100, temperature=0.8):
    """
    Legacy generate function for backward compatibility.
    Delegates to generate_base without mirror integration.
    """
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


def get_user_input_hidden(model, sp_tokenizer, device, conversation_tokens, new_input_tokens):
    """
    Get hidden states for user input (for comparison with expectation).
    """
    full_tokens = conversation_tokens + new_input_tokens
    context_tokens = full_tokens[-(config.pretrain_max_seq_len):]
    input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        _, hidden_states = model(input_ids, return_hidden=True)
    
    # Return only the hidden states corresponding to new input
    # This is the part that represents the user's message
    new_input_len = len(new_input_tokens)
    if new_input_len > 0 and hidden_states.shape[1] >= new_input_len:
        return hidden_states[:, -new_input_len:, :]
    return hidden_states


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
    
    # Mirror status
    use_mirror = config.USE_MIRROR_MODULE and MIRROR_AVAILABLE
    if use_mirror:
        print("🪞 Mirror Module is ACTIVE - predictive reflection enabled.")
        
        # Initialize embedding swap if configured
        swap_method = getattr(config, 'MIRROR_SWAP_METHOD', 'text')
        if swap_method == "embedding":
            print("   Initializing embedding-space swap vector...")
            mirror_integration.initialize_embedding_swap(model, sp, device)
            if mirror_integration._swap_initialized:
                print(f"   ✅ Embedding swap ready (method: {mirror_integration.swap_method})")
            else:
                print(f"   ⚠️ Fallback to text-based swap")
        else:
            print(f"   Using text-based swap (regex)")
    else:
        print("Mirror Module is disabled.")

    system_prompt = "Tvé jméno je Ája. Jsi prototyp BioCortexAI."
    prompt_formatted = f"system: {system_prompt} "
    conversation_tokens = [sp.bos_id()] + sp.encode_as_ids(prompt_formatted)
    
    # For mirror integration
    previous_user_hidden = None

    print("\nInput text (or 'exit' for end).")
    try:
        while True:
            prompt = input("User: ")
            if prompt.lower() in ["konec", "exit", "quit"]:
                break

            prompt_formatted = f"user: {prompt} model:"
            new_tokens = sp.encode_as_ids(prompt_formatted)
            
            if use_mirror:
                # Get hidden states for current user input
                current_user_hidden = get_user_input_hidden(
                    model, sp, device, conversation_tokens, new_tokens
                )
                
                # Generate with mirror prediction loop
                result = generate_with_mirror(
                    model, sp, device,
                    conversation_tokens,
                    new_tokens,
                    previous_user_hidden=previous_user_hidden,
                    max_new_tokens=100,
                    temperature=0.8
                )
                
                response = result["response"]
                hormones = result["hormones"]
                sentiment = result["sentiment"]
                conversation_tokens = result["updated_tokens"]
                prediction_result = result["prediction_result"]
                
                # Store current user hidden for next turn comparison
                previous_user_hidden = current_user_hidden
                
                print(f"Model: {response}")
                
                # Show prediction result
                if prediction_result:
                    quality_emoji = {
                        "good": "✅",
                        "neutral": "➖",
                        "bad": "❌"
                    }.get(prediction_result.quality, "❓")
                    print(f"[PREDICTION] {quality_emoji} Error: {prediction_result.prediction_error:.3f} | "
                          f"Cos.sim: {prediction_result.cosine_similarity:.3f} | Quality: {prediction_result.quality}")
                
                # Show hormone state
                if hormones:
                    mood_emoji, mood_color = get_mood_visualization(hormones)
                    hormone_str = " | ".join([f"{k.capitalize()}: {v:.2f}" for k, v in hormones.items()])
                    sentiment_str = f"Sentiment: {sentiment:.2f}" if sentiment is not None else "Sentiment: N/A"
                    print(f"[MODEL MOOD] {mood_emoji} | {hormone_str} | {sentiment_str}")
                
                # Show mirror stats
                if result.get("mirror_stats"):
                    stats = result["mirror_stats"]
                    print(f"[MIRROR STATS] Total: {stats['total_predictions']} | "
                          f"Good: {stats['good_predictions']} | "
                          f"Bad: {stats['bad_predictions']} | "
                          f"Accuracy: {stats['prediction_accuracy']:.1%}")
            
            else:
                # Legacy generation without mirror
                input_for_generate = conversation_tokens + new_tokens
                response, hormones, sentiment, updated_tokens = generate(
                    model, sp, device, input_for_generate, new_input_tokens=new_tokens
                )
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
        
        # Print final mirror statistics
        if use_mirror:
            stats = mirror_integration.get_stats()
            print(f"\n=== Final Mirror Statistics ===")
            print(f"Total predictions: {stats['total_predictions']}")
            print(f"Good predictions: {stats['good_predictions']}")
            print(f"Bad predictions: {stats['bad_predictions']}")
            print(f"Prediction accuracy: {stats['prediction_accuracy']:.1%}")

if __name__ == "__main__":
    main()