# BioCortexAI

![Verze](https://img.shields.io/badge/version-1.0--beta-blue)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Licence](https://img.shields.io/badge/license-CC--BY--NC--4.0-lightgrey)

**BioCortexAI je hybridní framework pro stavové jazykové modely, který kombinuje standardní Transformer architekturu s biologicky inspirovanou modulační vrstvou nazvanou "PlantNet".**

Tento přístup umožňuje modelu udržovat si vnitřní stav ("náladu"), který se dynamicky mění na základě interakcí a ovlivňuje jeho reakce v reálném čase. Výsledkem je AI, která je méně mechanická a více kontextuálně citlivá.

---

## Klíčové vlastnosti

- **Hybridní architektura**: Spojení výkonného LLM s dynamickou modulační sítí.
- **Vnitřní stav (Nálada)**: Modelovaný pomocí systému "hormonů" (dopamin, serotonin, kortizol, oxytocin), které ovlivňují výpočetní proces.
- **Tři úrovně učení**: Krátkodobé reakce, střednědobá asociativní paměť a dlouhodobá adaptace "osobnosti".
- **Modulární a konfigurovatelný**: Všechny parametry, od architektury modelu po strukturu PlantNet, jsou v `config.py`.
- **Kompletní workflow**: Obsahuje skripty pro přípravu dat, pre-training, fine-tuning, export a interaktivní chat.
- **Výpočetně efektivní**: PlantNet se adaptuje bez zpětné propagace, pouze pomocí jednoduchých lokálních pravidel.

## Jak to funguje?

Architektura funguje v neustálé zpětnovazební smyčce:

1.  **Modulace LLM**: PlantNet poskytne svůj aktuální hormonální stav. Tyto "hormony" v reálném čase upraví výpočty v `Attention` a `TransformerBlock` vrstvách LLM.
2.  **Generování odpovědi**: Modulovaný LLM vygeneruje odpověď.
3.  **Zpětná vazba do PlantNet**: Výstup LLM (`logits`, `hidden_states`) a text uživatele jsou analyzovány. Z nich se vypočítají signály (entropie, změna tématu, sentiment), které aktualizují stav PlantNet.

Tento cyklus se opakuje, což vede k neustále se vyvíjejícímu chování modelu.

## Struktura projektu

```
/biocortex_ai
|-- config.py               # Centrální konfigurace pro vše
|-- model.py                # Definice architektury Transformeru
|-- plant_net.py            # Definice architektury Rostlinné sítě
|-- sentiment_analyzer.py   # Síť pro analýzu sentimentu
|-- install_dependencies.py   # Skript pro instalaci závislostí
|-- pretrain.py             # Skript pro předtrénink základního modelu
|-- finetune.py             # Skript pro dolaďování (fine-tuning)
|-- export_model.py         # Skript pro export natrénovaného modelu
|-- generate.py             # Skript pro CLI interakci s modelem
|-- chat_UI.py              # Skript pro Gradio chat UI
|
|-- nastroje_pro_data/      # Skripty pro přípravu dat
|   |-- preprocess_corpus.py
|   |-- prepare_tokenizer.py
|   +-- chunk_corpus.py
|
|-- data/
|   |-- raw_data/           # Místo pro syrové .txt soubory
|   +-- CZ_QA_MIKRO.txt     # Ukázkový dataset pro fine-tuning
|
+-- checkpoints/
    |-- base_model/         # Sem se ukládá předtrénovaný model
    +-- finetuned_model/    # Sem se ukládá doladěný model
```

## Instalace

1.  **Naklonujte repozitář:**
    ```bash
    git clone https://github.com/VASE_JMENO/BioCortexAI.git
    cd BioCortexAI
    ```
2.  **(Doporučeno) Vytvořte a aktivujte virtuální prostředí:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Pro Linux/macOS
    # venv\Scripts\activate    # Pro Windows
    ```
3.  **Nainstalujte všechny závislosti:**
    ```bash
    python install_dependencies.py
    ```
    *Poznámka: Při prvním spuštění dojde k automatickému stažení modelu pro analýzu sentimentu `cardiffnlp/twitter-xlm-roberta-base-sentiment` (cca 1.1 GB).*

## Pracovní postup

Projekt má jasně definované fáze. Postupujte podle nich.

### 1. Příprava dat
1.  Umístěte své `.txt` soubory do `data/raw_data/`.
2.  Spusťte skripty v adresáři `nastroje_pro_data/` v tomto pořadí:
    ```bash
    python nastroje_pro_data/preprocess_corpus.py
    python nastroje_pro_data/prepare_tokenizer.py
    python nastroje_pro_data/chunk_corpus.py
    ```

### 2. Trénink modelu
1.  **Pre-training**: `python pretrain.py`
2.  **Fine-tuning**: `python finetune.py`

### 3. Export modelu
Zabalte finální model do jednoho souboru pro snadnou distribuci.
```bash
python export_model.py --input checkpoints/finetuned_model/latest_checkpoint.pt --output biocortex_model.pth
```

### 4. Interakce s modelem
Spusťte interaktivní webové rozhraní pro chat.
```bash
python chat_UI.py
```
*Poznámka: Skript `chat_UI.py` ve výchozím stavu hledá model `finetuned_model.pth`. Stav PlantNet se ukládá do `plant_net_state.json`.*

## Budoucí vývoj
- Experimenty s různými topologiemi a osobnostmi v `PLANT_TISSUE_MAP`.
- Integrace dalších "hormonů" pro komplexnější vnitřní stavy.
- Optimalizace výkonu a trénink větších modelů.
- Vylepšení a rozšíření `AssociativeMemory`.

## Jak přispět
Příspěvky jsou vítány! Pokud máte nápad na vylepšení nebo jste našli chybu, otevřete prosím "Issue" nebo pošlete "Pull Request".

## Licence
Tento projekt je licencován pod **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
Můžete volně sdílet a upravovat pro nekomerční účely za podmínky uvedení autora.
- **Plné znění licence**: [https://creativecommons.org/licenses/by-nc/4.0/legalcode](https://creativecommons.org/licenses/by-nc/4.0/legalcode)