# BioCortexAI

![Version](https://img.shields.io/badge/version-1.0--beta-blue)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-CC--BY--NC--4.0-lightgrey)

**BioCortexAI is a hybrid framework for stateful language models, combining a standard Transformer architecture with a biologically-inspired modulation layer called "PlantNet."**

What if a language model wasn't just a tool, but a dynamic system with an internal state? What if it could remember, adapt, and have its "mood" influence its responses?
BioCortexAI is an open-source framework that attempts to answer these questions.
By integrating a standard Transformer architecture with a biologically-inspired modulation layer called "PlantNet," we give the model a "mood" and "memory" through a simulated hormonal system that reacts to the flow of conversation.
Our goal is not just to generate text, but to explore how to create an AI that is more contextually aware, less mechanical, and whose behavior evolves over time.
This repository contains a complete toolkit to train, experiment with, and interact with your own stateful model. Welcome to the world of BioCortexAI.

This approach allows the model to maintain an internal state ("mood") that dynamically changes based on interactions, influencing its responses in real-time. The result is an AI that is less mechanical and more contextually aware.

---

## Key Features

- **Hybrid Architecture**: Merges a powerful LLM with a dynamic modulation network.
- **Internal State (Mood)**: Modeled using a system of "hormones" (dopamine, serotonin, cortisol, oxytocin) that affect the computational process.
- **Three Levels of Learning**: Short-term reactions, medium-term associative memory, and long-term "personality" adaptation.
- **Modular and Configurable**: All parameters, from model architecture to the PlantNet structure, are defined in `config.py`.
- **Complete Workflow**: Includes scripts for data preparation, pre-training, fine-tuning, exporting, and interactive chat.
- **Computationally Efficient**: PlantNet adapts without backpropagation, using only simple, local rules.

## How It Works

The architecture operates in a continuous feedback loop:

1.  **LLM Modulation**: PlantNet provides its current hormonal state. These "hormones" modify the calculations within the LLM's `Attention` and `TransformerBlock` layers in real-time.
2.  **Response Generation**: The modulated LLM generates a response.
3.  **Feedback to PlantNet**: The LLM's output (`logits`, `hidden_states`) and the user's text are analyzed. Signals (entropy, topic deviation, sentiment) are calculated from them to update the PlantNet's state.

This cycle repeats, leading to the model's ever-evolving behavior.

## Project Structure
```
/biocortex_ai
|-- config.py               # Central configuration for everything
|-- model.py                # Transformer architecture definition
|-- plant_net.py            # PlantNet architecture definition
|-- sentiment_analyzer.py   # Sentiment analysis network
|-- install_dependencies.py   # Script to install dependencies
|-- pretrain.py             # Script for pre-training the base model
|-- finetune.py             # Script for fine-tuning
|-- export_model.py         # Script for exporting a trained model
|-- generate.py             # Script for CLI interaction with the model
|-- chat_UI.py              # Script for the Gradio chat UI
|
|-- data_tools/             # Scripts for data preparation
|   |-- preprocess_corpus.py
|   |-- prepare_tokenizer.py
|   +-- chunk_corpus.py
|
|-- data/
|   |-- raw_data/           # Directory for raw .txt files
|   +-- sample_finetune.txt # Sample dataset for fine-tuning
|
+-- checkpoints/
    |-- base_model/         # Pre-trained model is saved here
    +-- finetuned_model/    # Fine-tuned model is saved here
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/BioCortexAI.git
    cd BioCortexAI
    ```
2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/macOS
    # venv\Scripts\activate    # For Windows
    ```
3.  **Install all dependencies:**
    ```bash
    python install_dependencies.py
    ```
    *Note: The first time you run a script, the sentiment analysis model `cardiffnlp/twitter-xlm-roberta-base-sentiment` (approx. 1.1 GB) will be downloaded automatically.*

## Workflow

The project has clearly defined phases. Follow them in order.

### 1. Data Preparation
1.  Place your `.txt` files in the `data/raw_data/` directory.
2.  Run the scripts from the `data_tools/` directory sequentially:
    ```bash
    python data_tools/preprocess_corpus.py
    python data_tools/prepare_tokenizer.py
    python data_tools/chunk_corpus.py
    ```

### 2. Model Training
1.  **Pre-training**: `python pretrain.py`
2.  **Fine-tuning**: `python finetune.py`

### 3. Model Export
Package the final model into a single, self-contained file.
```bash
python export_model.py --input checkpoints/finetuned_model/latest_checkpoint.pt --output biocortex_model.pth
```

### 4. Interaction
Launch the interactive web UI to chat with your model.
```bash
python chat_UI.py
```
*Note: By default, `chat_UI.py` looks for a model named `finetuned_model.pth`. The PlantNet state is saved to `plant_net_state.json`.*

## Future Work
- Experimenting with different topologies and personalities in `PLANT_TISSUE_MAP`.
- Integrating additional "hormones" for more complex internal states.
- Performance optimization and training larger models.
- Improving and extending the `AssociativeMemory`.

## Contributing
Contributions are welcome! If you have an idea for an improvement or have found a bug, please open an issue or submit a pull request.

## License
This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.
You are free to share and adapt this work for non-commercial purposes, provided you give appropriate credit.
- **Full License Text**: [https://creativecommons.org/licenses/by-nc/4.0/legalcode](https://creativecommons.org/licenses/by-nc/4.0/legalcode)
