import torch
import sentencepiece as spm
import gradio as gr
import pandas as pd
import config
from model import Transformer
from plant_net import DistributedPlantNetwork

MODEL_PATH = "finetuned_model.pth"
PLANT_STATE_PATH = config.PLANT_NET_STATE_FILE

# === Load model ===
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
conversation_tokens = [sp.bos_id()]  # conv. history

# Mood and sentiment history
hormone_history = {"dopamin": [], "serotonin": [], "kortizol": [], "oxytocin": []}
sentiment_history = []

# === Function generate ===
@torch.no_grad()
def chat_fn(user_input, history):
    global conversation_tokens, hormone_history, sentiment_history

    prompt_formatted = f"user: {user_input} model:"
    new_tokens = sp.encode_as_ids(prompt_formatted)
    conversation_tokens = conversation_tokens + new_tokens
    decoded_string = sp.decode_ids(conversation_tokens)
    
    input_ids = torch.tensor([conversation_tokens], dtype=torch.long, device=device)
    initial_len = len(conversation_tokens)

    sentiment = None
    for _ in range(100):  # max_new_tokens
        input_cond = input_ids[:, -config.pretrain_max_seq_len:]
        logits, hidden_states = model(input_cond, return_hidden=True)

        if model.use_plant_net and model.plant_layer:
            sentiment = model.plant_layer.update_state(logits, hidden_states, (decoded_string[-100:]))

        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == sp.eos_id():
            break

        input_ids = torch.cat([input_ids, next_id], dim=1)

    all_tokens = input_ids[0].tolist()
    new_generated = all_tokens[initial_len:]
    response = sp.decode(new_generated).strip()
    conversation_tokens = all_tokens

    hormones = model.plant_layer.get_global_hormones() if model.use_plant_net else None
    if hormones:
        for k in hormone_history.keys():
            hormone_history[k].append(hormones[k])
    sentiment_history.append(sentiment if sentiment is not None else 0.0)

    mood_str = " | ".join([f"{k.capitalize()}: {v:.2f}" for k, v in hormones.items()]) if hormones else "N/A"
    sentiment_str = f"Sentiment: {sentiment:.2f}" if sentiment is not None else "Sentiment: N/A"

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": f"{response}\n\n[Nálada] {mood_str} | {sentiment_str}"})

    chart_data = {
        "step": list(range(len(hormone_history["dopamin"]))),
        "dopamin": hormone_history["dopamin"],
        "serotonin": hormone_history["serotonin"],
        "kortizol": hormone_history["kortizol"],
        "oxytocin": hormone_history["oxytocin"],
        "sentiment": sentiment_history,
    }
    df_wide = pd.DataFrame(chart_data)

    chart_df = df_wide.melt(
        id_vars=["step"],
        value_vars=["dopamin", "serotonin", "kortizol", "oxytocin", "sentiment"],
        var_name="variable",
        value_name="value"
    )

    return history, chart_df


# === Function reset textbox ===
def reset_msg():
    return ""


# === Function manual PlantNet save ===
def save_plant_state():
    if model.use_plant_net and model.plant_layer:
        model.plant_layer.network.save(PLANT_STATE_PATH)
        return f"✅ Stav rostlinné sítě uložen do: {PLANT_STATE_PATH}"
    return "⚠️ PlantNet není aktivní."


# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🌱 BioCortexAI Chat (Ája cz 13M model)")

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Napiš zprávu")

    with gr.Row():
        clear = gr.Button("🧹 Vymazat chat")
        save_btn = gr.Button("💾 Uložit PlantNet")

    status = gr.Textbox(label="Status ukládání", interactive=False)

    with gr.Row():
        lineplot = gr.LinePlot(
            x="step",
            y="value",
            color="variable",
            x_label="Krok",
            y_label="Hodnota",
            title="Vývoj hormonů a sentimentu",
            overlay_point=False
        )

    msg.submit(chat_fn, [msg, chatbot], [chatbot, lineplot])
    msg.submit(reset_msg, None, msg, queue=False)

    clear.click(lambda: ([], pd.DataFrame(columns=["step","variable","value"])),
                None, [chatbot, lineplot])

    save_btn.click(save_plant_state, None, status)

# === Launch with autosave ===
try:
    demo.launch()
finally:
    if model.use_plant_net and model.plant_layer:
        model.plant_layer.network.save(PLANT_STATE_PATH)
        print(f"\n✅ Stav rostlinné sítě uložen do: {PLANT_STATE_PATH}")
