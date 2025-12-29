# chat_ui.py
# -*- coding: utf-8 -*-
"""
BioCortexAI Chat UI with Mirror Integration
============================================
Gradio-based chat interface with hormone visualization and mirror prediction display.

(c) 2025 OpenTechLab Jablonec nad Nisou s.r.o.
Author: Michal Seidl
"""

import torch
import sentencepiece as spm
import gradio as gr
import pandas as pd
import config
from model import Transformer
from plant_net import DistributedPlantNetwork
from generate import generate_with_mirror, generate, get_mood_visualization, get_user_input_hidden

# Import Mirror Integration
try:
    from mirror_integration import mirror_integration
    MIRROR_AVAILABLE = True
except ImportError:
    MIRROR_AVAILABLE = False
    print("Warning: Mirror integration not available")

MODEL_PATH = "finetuned_model.pth"
PLANT_STATE_PATH = config.PLANT_NET_STATE_FILE


# === Load model and tokenizer ===
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_data = torch.load(MODEL_PATH, map_location=device)
    cfg = model_data["config"]

    model = Transformer(
        vocab_size=cfg["vocab_size"],
        dim=cfg["embedding_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        ff_multiplier=cfg["ff_multiplier"],
        dropout=0.0
    ).to(device)
    model.load_state_dict(model_data["model_state_dict"])

    if model.use_plant_net:
        if "initial_plant_state" in model_data and model_data["initial_plant_state"] is not None:
            model.plant_layer.network = DistributedPlantNetwork.from_dict_static(
                model_data["initial_plant_state"]
            )

    sp = spm.SentencePieceProcessor()
    sp.load(f"{config.model_prefix}.model")
    return model, sp, device


model, sp, device = load_model()
conversation_tokens = [sp.bos_id()]  # conversation history

# Initialize embedding swap if configured
if config.USE_MIRROR_MODULE and MIRROR_AVAILABLE:
    swap_method = getattr(config, 'MIRROR_SWAP_METHOD', 'text')
    if swap_method == "embedding":
        print("Initializing embedding-space swap vector...")
        mirror_integration.initialize_embedding_swap(model, sp, device)

# History for hormones, sentiment, and prediction error
hormone_history = {"dopamin": [], "serotonin": [], "kortizol": [], "oxytocin": []}
prediction_history = {"error": [], "quality": []}
previous_user_hidden = None  # For mirror comparison


# === Chat function with mirror integration ===
@torch.no_grad()
def chat_fn(user_input, history):
    global conversation_tokens, hormone_history, prediction_history, previous_user_hidden

    prompt_formatted = f"user: {user_input} model:"
    new_tokens = sp.encode_as_ids(prompt_formatted)
    
    use_mirror = config.USE_MIRROR_MODULE and MIRROR_AVAILABLE
    
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
        conversation_tokens = result["updated_tokens"]
        prediction_result = result["prediction_result"]
        
        # Store current user hidden for next turn
        previous_user_hidden = current_user_hidden
        
        # Build response with prediction info
        mood_emoji, _ = get_mood_visualization(hormones) if hormones else ("😐", "#F5F5F5")
        
        # Add prediction quality indicator
        if prediction_result:
            pred_emoji = {"good": "✅", "neutral": "➖", "bad": "❌"}.get(prediction_result.quality, "")
            prediction_history["error"].append(prediction_result.prediction_error)
            prediction_history["quality"].append(
                1.0 if prediction_result.quality == "good" else 
                0.5 if prediction_result.quality == "neutral" else 0.0
            )
            final_response = f"{mood_emoji}{pred_emoji}|{response.strip()}"
        else:
            final_response = f"{mood_emoji}|{response.strip()}"
            # Pad prediction history when no prediction available
            if prediction_history["error"]:
                prediction_history["error"].append(prediction_history["error"][-1] if prediction_history["error"] else 0.5)
                prediction_history["quality"].append(prediction_history["quality"][-1] if prediction_history["quality"] else 0.5)
    
    else:
        # Legacy generation without mirror
        input_for_generate = conversation_tokens + new_tokens
        response, hormones, sentiment, updated_tokens = generate(
            model, sp, device, input_for_generate, new_input_tokens=new_tokens
        )
        conversation_tokens = updated_tokens
        
        mood_emoji, _ = get_mood_visualization(hormones) if hormones else ("😐", "#F5F5F5")
        final_response = f"{mood_emoji}|{response.strip()}"

    # Update hormone history
    if hormones:
        for k in hormone_history.keys():
            hormone_history[k].append(hormones[k])

    # Add to chat history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": final_response})

    # Prepare hormone chart data
    hormone_chart_data = {
        "step": list(range(len(hormone_history["dopamin"]))),
        "dopamin": hormone_history["dopamin"],
        "serotonin": hormone_history["serotonin"],
        "kortizol": hormone_history["kortizol"],
        "oxytocin": hormone_history["oxytocin"],
    }
    df_hormones = pd.DataFrame(hormone_chart_data)
    hormone_chart_df = df_hormones.melt(
        id_vars=["step"],
        value_vars=["dopamin", "serotonin", "kortizol", "oxytocin"],
        var_name="hormone",
        value_name="value"
    )

    # Prepare prediction chart data
    if prediction_history["error"] and len(prediction_history["error"]) > 0:
        prediction_chart_data = {
            "step": list(range(len(prediction_history["error"]))),
            "prediction_error": prediction_history["error"],
            "prediction_quality": prediction_history["quality"],
        }
        df_prediction = pd.DataFrame(prediction_chart_data)
        prediction_chart_df = df_prediction.melt(
            id_vars=["step"],
            value_vars=["prediction_error", "prediction_quality"],
            var_name="metric",
            value_name="value"
        )
    else:
        prediction_chart_df = pd.DataFrame(columns=["step", "metric", "value"])

    # Get mirror stats
    mirror_stats_text = ""
    if use_mirror:
        stats = mirror_integration.get_stats()
        mirror_stats_text = (
            f"🪞 Mirror Stats: "
            f"Total: {stats['total_predictions']} | "
            f"Good: {stats['good_predictions']} | "
            f"Bad: {stats['bad_predictions']} | "
            f"Accuracy: {stats['prediction_accuracy']:.1%}"
        )

    return history, hormone_chart_df, prediction_chart_df, mirror_stats_text


# === Reset and save functions ===
def reset_msg():
    return ""


def reset_all():
    global conversation_tokens, hormone_history, prediction_history, previous_user_hidden
    conversation_tokens = [sp.bos_id()]
    hormone_history = {"dopamin": [], "serotonin": [], "kortizol": [], "oxytocin": []}
    prediction_history = {"error": [], "quality": []}
    previous_user_hidden = None
    
    if MIRROR_AVAILABLE:
        mirror_integration.clear_expectation()
        mirror_integration.total_predictions = 0
        mirror_integration.good_predictions = 0
        mirror_integration.bad_predictions = 0
    
    empty_hormone_df = pd.DataFrame(columns=["step", "hormone", "value"])
    empty_prediction_df = pd.DataFrame(columns=["step", "metric", "value"])
    return [], empty_hormone_df, empty_prediction_df, ""


def save_plant_state():
    if model.use_plant_net and model.plant_layer:
        model.plant_layer.network.save(PLANT_STATE_PATH)
        return f"✅ Plant network state saved to: {PLANT_STATE_PATH}"
    return "⚠️ PlantNet is not active."


# === Gradio UI ===
with gr.Blocks(title="BioCortexAI Chat") as demo:
    gr.Markdown("## 🌱🪞 BioCortexAI Chat (with Mirror Integration)")
    gr.Markdown("*Model s prediktivním digitálním zrcadlem – anticipuje reakce uživatele*")

    chatbot = gr.Chatbot(type="messages", height=400)
    msg = gr.Textbox(label="Napiš zprávu", placeholder="Napište svou zprávu...")

    with gr.Row():
        clear = gr.Button("🧹 Reset vše")
        save_btn = gr.Button("💾 Uložit PlantNet")

    status = gr.Textbox(label="Status", interactive=False)
    mirror_stats = gr.Textbox(label="🪞 Mirror Statistics", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧪 Hormonální stav")
            hormone_plot = gr.LinePlot(
                x="step",
                y="value",
                color="hormone",
                x_title="Krok",
                y_title="Hodnota",
                title="Vývoj hormonů v čase",
                overlay_point=False,
                height=250
            )
        with gr.Column():
            gr.Markdown("### 🔮 Predikční přesnost zrcadla")
            prediction_plot = gr.LinePlot(
                x="step",
                y="value",
                color="metric",
                x_title="Krok",
                y_title="Hodnota",
                title="Prediction Error & Quality",
                overlay_point=False,
                height=250
            )

    # Event handlers
    msg.submit(
        chat_fn, 
        [msg, chatbot], 
        [chatbot, hormone_plot, prediction_plot, mirror_stats]
    )
    msg.submit(reset_msg, None, msg, queue=False)

    clear.click(
        reset_all, 
        None, 
        [chatbot, hormone_plot, prediction_plot, mirror_stats]
    )

    save_btn.click(save_plant_state, None, status)

    # Footer
    gr.Markdown("""
    ---
    **Legenda:**
    - 🪞 **Mirror**: Model predikuje, jak uživatel odpoví, a porovnává s realitou
    - ✅ **Good**: Model správně anticipoval reakci uživatele
    - ❌ **Bad**: Uživatel odpověděl jinak, než model očekával (→ učení)
    - Hormony ovlivňují generování odpovědí v reálném čase
    
    *(c) 2025 OpenTechLab Jablonec nad Nisou s.r.o.*
    """)


# === Launch with autosave ===
if __name__ == "__main__":
    try:
        demo.launch()
    finally:
        if model.use_plant_net and model.plant_layer:
            model.plant_layer.network.save(PLANT_STATE_PATH)
            print(f"\n✅ Plant network state saved to: {PLANT_STATE_PATH}")
