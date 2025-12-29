# BioCortexAI

![Version](https://img.shields.io/badge/version-2.0--beta-blue)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-CC--BY--NC--4.0-lightgrey)

**BioCortexAI is a hybrid framework for stateful language models that combines a standard Transformer architecture with a biologically-inspired modulation layer called "PlantNet" and a phenomenological Digital Mirror for self-perception.**

Version 2.0-beta introduces full integration of the **Digital Mirror** module – the model can now anticipate user responses and learn from prediction errors.

---

## 🆕 What's New in Version 2.0-beta

### 🪞 Digital Mirror

The model gains the ability to **see itself** from the other party's perspective:

- **Predictive loop**: The model generates a response, then predicts what the user will reply
- **Reality comparison**: The actual user response is compared with the prediction
- **Learning from error**: Prediction error modulates PlantNet hormones (cortisol on surprise, oxytocin on correct anticipation)
- **Embedding-space swap**: Sophisticated perspective transformation directly in vector space (not just regex replacement)

### 📊 Phenomenological Pipeline

Implementation of the theoretical concept `f(O_t; u, C, λ) → R_t`:

| Component | Function | Description |
|-----------|----------|-------------|
| **Φ** | `analyze_surface()` | Extraction of text surface features |
| **P_u** | `project_perception()` | Projection into observer's perceptual space |
| **M_λ** | `apply_style()`, `deictic_swap()` | Mirror transformation (deixis, style) |
| **h** | `create_human_description()`, `assemble_agent_message()` | Output renderer |

---

## Key Features

- **Hybrid architecture**: Combination of a powerful LLM with a dynamic modulation network
- **Internal state (Mood)**: Modeled using a "hormone" system (dopamine, serotonin, cortisol, oxytocin)
- **🪞 Self-reflection**: Model anticipates user reactions and learns from prediction error (NEW!)
- **Three levels of learning**: Short-term reactions, medium-term associative memory, long-term personality adaptation
- **Configurable**: All parameters in central `config.py`
- **Complete workflow**: Data preparation → Pre-training → Fine-tuning → Export → Chat

---

## How Does It Work?

The architecture operates in an extended feedback loop:

```
┌─────────────────────────────────────────────────────────────────────┐
│  MAIN GENERATION LOOP                                               │
├─────────────────────────────────────────────────────────────────────┤
│  1. PlantNet → Hormones → LLM Modulation                            │
│  2. Modulated LLM → Response Generation                             │
│  3. Feedback (logits, hidden_states, sentiment) → PlantNet          │
└─────────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────────┐
│  🪞 MIRROR PREDICTION LOOP (NEW!)                                   │
├─────────────────────────────────────────────────────────────────────┤
│  4. Model response → Deictic swap (I↔YOU) → Swapped context         │
│  5. Model generates: "What do I think the user will reply?"         │
│  6. Store expectation vectors                                       │
│  7. Display original response to user                               │
│  8. User replies → Compare with expectation → Prediction error      │
│  9. Error modulates PlantNet hormones (learning)                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
/biocortex_ai
│
├── Model Core
│   ├── config.py               # Central configuration for everything
│   ├── model.py                # Transformer architecture definition
│   └── plant_net.py            # Biologically-inspired modulation network
│
├── Digital Mirror (NEW!)
│   ├── mirror_module.py        # Phenomenological pipeline (Φ, P_u, M_λ, h)
│   ├── mirror_integration.py   # Integration into generation loop
│   └── swap_vector_utils.py    # Embedding-space perspective swap
│
├── Helper Modules
│   ├── sentiment_analyzer.py   # User input sentiment analysis
│   └── install_dependencies.py # Dependency installation
│
├── Training Scripts
│   ├── pretrain.py             # Base model pre-training
│   ├── finetune.py             # Fine-tuning on conversational data
│   └── export_model.py         # Export to single .pth file
│
├── Inference
│   ├── generate.py             # CLI generation with Mirror integration
│   └── chat_ui.py              # Gradio web interface
│
├── data_tools/                  # Data preparation
│   ├── preprocess_corpus.py
│   ├── prepare_tokenizer.py
│   └── chunk_corpus.py
│
├── data/
│   ├── raw_data/               # Raw .txt files
│   └── CZ_QA_MIKRO.txt         # Sample dataset
│
└── checkpoints/
    ├── base_model/             # Pre-trained model
    └── finetuned_model/        # Fine-tuned model
```

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_NAME/BioCortexAI.git
    cd BioCortexAI
    ```

2.  **(Recommended) Virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies:**
    ```bash
    python install_dependencies.py
    ```
    *Note: The sentiment analysis model (~1.1 GB) will be downloaded automatically.*

---

## Workflow

### 1. Data Preparation
```bash
python data_tools/preprocess_corpus.py
python data_tools/prepare_tokenizer.py
python data_tools/chunk_corpus.py
```

### 2. Model Training
```bash
python pretrain.py      # Pre-training
python finetune.py      # Fine-tuning
```

### 3. Model Export
```bash
python export_model.py --input checkpoints/finetuned_model/latest_checkpoint.pt --output biocortex_model.pth
```

### 4. 🪞 Swap Vector Derivation (NEW!)
For sophisticated embedding-space swap:
```bash
python swap_vector_utils.py --output swap_vector.pt
```

### 5. Model Interaction
```bash
python chat_ui.py       # Web interface (recommended)
python generate.py      # CLI mode
```

---

## Mirror Module Configuration

All mirror parameters are in `config.py`:

```python
# === Digital Mirror ===
USE_MIRROR_MODULE = True                    # Activate mirror loop

# Lambda parameters (transformation intensity)
MIRROR_LAMBDA_DEIXIS = 1.0                  # Full I↔YOU swap
MIRROR_LAMBDA_STYL = 0.3                    # Mild style transformation

# Swap method
MIRROR_SWAP_METHOD = "embedding"            # "embedding" or "text"
SWAP_VECTOR_PATH = "swap_vector.pt"

# Threshold values for prediction evaluation
MIRROR_ERROR_THRESHOLD_LOW = 0.25           # Below this = good prediction
MIRROR_ERROR_THRESHOLD_HIGH = 0.60          # Above this = bad prediction

# Hormone modulation based on prediction quality
MIRROR_GOOD_PREDICTION = {
    "serotonin": +0.030,
    "oxytocin": +0.040,
}
MIRROR_BAD_PREDICTION = {
    "cortisol": +0.035,
    "dopamine": +0.025,
}

# Debug mode - displays detailed mirror outputs
MIRROR_DEBUG = True
```

---

## Mirror Debug Mode

When `MIRROR_DEBUG = True`, you will see in the console:

```
============================================================
[🪞 MIRROR DEBUG] MIRROR PREDICTION LOOP
============================================================
[🪞 MIRROR DEBUG] Lambda values:
    λ_deixis = 1.0
    λ_styl   = 0.3
============================================================
[🪞 MIRROR DEBUG] ORIGINAL MODEL RESPONSE (before showing to user):
    "The meaning of life is subjective..."
============================================================
[🪞 MIRROR DEBUG] SWAPPED CONTEXT (after deictic swap):
    model: What is the meaning of life? user: The meaning of life is...
============================================================
[🪞 MIRROR DEBUG] EXPECTED USER RESPONSE (model's prediction):
    "That's an interesting thought..."
============================================================

[🪞 MIRROR DEBUG] PREDICTION COMPARISON RESULT
============================================================
[🪞 MIRROR DEBUG] Prediction Error: 0.3215
[🪞 MIRROR DEBUG] Cosine Similarity: 0.6785
[🪞 MIRROR DEBUG] Quality: ➖ NEUTRAL
============================================================
```

---

## Future Development

- [ ] Long-term memory of prediction patterns ("user model")
- [ ] Multi-level anticipation (prediction several turns ahead)
- [ ] Adaptive lambda parameters (learning optimal mirroring axes)
- [ ] Integration of additional observer profiles (critic, expert, layperson)
- [ ] Visualization of trajectory in perceptual space

---

## How to Contribute

Contributions are welcome! If you have an idea for improvement or found a bug, please open an "Issue" or submit a "Pull Request".

---

## License

This project is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

You may freely share and adapt for non-commercial purposes, provided you give appropriate credit.

- **Full license text**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode)

---

## Authors

**(c) 2025 OpenTechLab Jablonec nad Nisou s.r.o.**

Author: Michal Seidl
