# BioCortexAI

![Verze](https://img.shields.io/badge/version-2.0--beta-blue)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Licence](https://img.shields.io/badge/license-CC--BY--NC--4.0-lightgrey)

**BioCortexAI je hybridní framework pro stavové jazykové modely, který kombinuje standardní Transformer architekturu s biologicky inspirovanou modulační vrstvou „PlantNet" a fenomenologickým digitálním zrcadlem pro sebepercepci.**

Verze 2.0-beta přináší plnou integraci **Digital Mirror** modulu – model nyní umí anticipovat reakce uživatele a učit se z predikční chyby.

---

## 🆕 Co je nového ve verzi 2.0-beta

### 🪞 Digital Mirror (Digitální zrcadlo)

Model získává schopnost **vidět sám sebe** z perspektivy druhé strany:

- **Prediktivní smyčka**: Model generuje odpověď, pak predikuje, co uživatel odpoví
- **Porovnání s realitou**: Skutečná odpověď uživatele se porovná s predikcí
- **Učení z chyby**: Predikční chyba moduluje PlantNet hormony (kortizol při překvapení, oxytocin při správné anticipaci)
- **Embedding-space swap**: Sofistikovaná perspektivní transformace přímo ve vektorovém prostoru (nejen regex nahrazení)

### 📊 Fenomenologický pipeline

Implementace teoretického konceptu `f(O_t; u, C, λ) → R_t`:

| Komponenta | Funkce | Popis |
|------------|--------|-------|
| **Φ** | `analyzuj_povrch()` | Extrakce povrchových rysů textu |
| **P_u** | `projektuj_vnimani()` | Projekce do percepčního prostoru pozorovatele |
| **M_λ** | `aplikuj_styl()`, `deikticky_swap()` | Zrcadlová transformace (deixis, styl) |
| **h** | `vytvor_lidsky_popis()`, `sestav_agent_zpravu()` | Renderer výstupu |

---

## Klíčové vlastnosti

- **Hybridní architektura**: Spojení výkonného LLM s dynamickou modulační sítí
- **Vnitřní stav (Nálada)**: Modelovaný pomocí systému „hormonů" (dopamin, serotonin, kortizol, oxytocin)
- **🪞 Sebereflexe**: Model anticipuje reakce uživatele a učí se z predikční chyby (NEW!)
- **Tři úrovně učení**: Krátkodobé reakce, střednědobá asociativní paměť, dlouhodobá adaptace osobnosti
- **Konfigurovatelný**: Všechny parametry v centrálním `config.py`
- **Kompletní workflow**: Příprava dat → Pre-training → Fine-tuning → Export → Chat

---

## Jak to funguje?

Architektura funguje v rozšířené zpětnovazební smyčce:

```
┌─────────────────────────────────────────────────────────────────────┐
│  HLAVNÍ GENEROVACÍ SMYČKA                                           │
├─────────────────────────────────────────────────────────────────────┤
│  1. PlantNet → Hormony → Modulace LLM                               │
│  2. Modulovaný LLM → Generování odpovědi                            │
│  3. Zpětná vazba (logits, hidden_states, sentiment) → PlantNet      │
└─────────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────────┐
│  🪞 ZRCADLOVÁ PREDIKČNÍ SMYČKA (NEW!)                               │
├─────────────────────────────────────────────────────────────────────┤
│  4. Odpověď modelu → Deictic swap (JÁ↔TY) → Swapped context         │
│  5. Model generuje: "Co si myslím, že uživatel odpoví?"             │
│  6. Uložení expectation vektorů                                     │
│  7. Zobrazení původní odpovědi uživateli                            │
│  8. Uživatel odpoví → Porovnání s expectation → Predikční chyba     │
│  9. Chyba moduluje PlantNet hormony (učení)                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Struktura projektu

```
/biocortex_ai
│
├── Jádro modelu
│   ├── config.py               # Centrální konfigurace pro vše
│   ├── model.py                # Definice architektury Transformeru
│   └── plant_net.py            # Biologicky inspirovaná modulační síť
│
├── Digital Mirror (NEW!)
│   ├── mirror_module.py        # Fenomenologický pipeline (Φ, P_u, M_λ, h)
│   ├── mirror_integration.py   # Integrace do generovací smyčky
│   └── swap_vector_utils.py    # Embedding-space perspektivní swap
│
├── Pomocné moduly
│   ├── sentiment_analyzer.py   # Analýza sentimentu uživatelského vstupu
│   └── install_dependencies.py # Instalace závislostí
│
├── Trénovací skripty
│   ├── pretrain.py             # Pre-training základního modelu
│   ├── finetune.py             # Fine-tuning na konverzačních datech
│   └── export_model.py         # Export do jednoho .pth souboru
│
├── Inference
│   ├── generate.py             # CLI generování s Mirror integrace
│   └── chat_ui.py              # Gradio webové rozhraní
│
├── nastroje_pro_data/          # Příprava dat
│   ├── preprocess_corpus.py
│   ├── prepare_tokenizer.py
│   └── chunk_corpus.py
│
├── data/
│   ├── raw_data/               # Syrové .txt soubory
│   └── CZ_QA_MIKRO.txt         # Ukázkový dataset
│
└── checkpoints/
    ├── base_model/             # Předtrénovaný model
    └── finetuned_model/        # Doladěný model
```

---

## Instalace

1.  **Naklonujte repozitář:**
    ```bash
    git clone https://github.com/VASE_JMENO/BioCortexAI.git
    cd BioCortexAI
    ```

2.  **(Doporučeno) Virtuální prostředí:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

3.  **Nainstalujte závislosti:**
    ```bash
    python install_dependencies.py
    ```
    *Pozn.: Automaticky se stáhne model pro sentiment analýzu (~1.1 GB).*

---

## Pracovní postup

### 1. Příprava dat
```bash
python nastroje_pro_data/preprocess_corpus.py
python nastroje_pro_data/prepare_tokenizer.py
python nastroje_pro_data/chunk_corpus.py
```

### 2. Trénink modelu
```bash
python pretrain.py      # Pre-training
python finetune.py      # Fine-tuning
```

### 3. Export modelu
```bash
python export_model.py --input checkpoints/finetuned_model/latest_checkpoint.pt --output biocortex_model.pth
```

### 4. 🪞 Odvození swap vektoru (NEW!)
Pro sofistikovaný embedding-space swap:
```bash
python swap_vector_utils.py --output swap_vector.pt
```

### 5. Interakce s modelem
```bash
python chat_ui.py       # Webové rozhraní (doporučeno)
python generate.py      # CLI mode
```

---

## Konfigurace Mirror Module

Všechny parametry zrcadla jsou v `config.py`:

```python
# === Digital Mirror ===
USE_MIRROR_MODULE = True                    # Aktivace zrcadlové smyčky

# Lambda parametry (intenzita transformací)
MIRROR_LAMBDA_DEIXIS = 1.0                  # Plný swap JÁ↔TY
MIRROR_LAMBDA_STYL = 0.3                    # Mírná stylová transformace

# Metoda swapu
MIRROR_SWAP_METHOD = "embedding"            # "embedding" nebo "text"
SWAP_VECTOR_PATH = "swap_vector.pt"

# Prahové hodnoty pro hodnocení predikce
MIRROR_ERROR_THRESHOLD_LOW = 0.25           # Pod tímto = dobrá predikce
MIRROR_ERROR_THRESHOLD_HIGH = 0.60          # Nad tímto = špatná predikce

# Modulace hormonů podle kvality predikce
MIRROR_GOOD_PREDICTION = {
    "serotonin": +0.030,
    "oxytocin": +0.040,
}
MIRROR_BAD_PREDICTION = {
    "kortizol": +0.035,
    "dopamin": +0.025,
}

# Debug mód - zobrazí detailní výstupy zrcadla
MIRROR_DEBUG = True
```

---

## Debug mód zrcadla

Při `MIRROR_DEBUG = True` uvidíte v konzoli:

```
============================================================
[🪞 MIRROR DEBUG] MIRROR PREDICTION LOOP
============================================================
[🪞 MIRROR DEBUG] Lambda values:
    λ_deixis = 1.0
    λ_styl   = 0.3
============================================================
[🪞 MIRROR DEBUG] ORIGINAL MODEL RESPONSE (before showing to user):
    "Smysl života je subjektivní..."
============================================================
[🪞 MIRROR DEBUG] SWAPPED CONTEXT (after deictic swap):
    model: Jaký je smysl života? user: Smysl života je...
============================================================
[🪞 MIRROR DEBUG] EXPECTED USER RESPONSE (model's prediction):
    "To je zajímavá myšlenka..."
============================================================

[🪞 MIRROR DEBUG] PREDICTION COMPARISON RESULT
============================================================
[🪞 MIRROR DEBUG] Prediction Error: 0.3215
[🪞 MIRROR DEBUG] Cosine Similarity: 0.6785
[🪞 MIRROR DEBUG] Quality: ➖ NEUTRAL
============================================================
```

---

## Budoucí vývoj

- [ ] Dlouhodobá paměť predikčních vzorců ("model uživatele")
- [ ] Víceúrovňová anticipace (predikce několika tahů dopředu)
- [ ] Adaptivní lambda parametry (učení optimálních os zrcadlení)
- [ ] Integrace dalších pozorovatelských profilů (kritik, expert, laik)
- [ ] Vizualizace trajektorie v percepčním prostoru

---

## Jak přispět

Příspěvky jsou vítány! Pokud máte nápad na vylepšení nebo jste našli chybu, otevřete prosím „Issue" nebo pošlete „Pull Request".

---

## Licence

Tento projekt je licencován pod **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

Můžete volně sdílet a upravovat pro nekomerční účely za podmínky uvedení autora.

- **Plné znění licence**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode)

---

## Autoři

**(c) 2025 OpenTechLab Jablonec nad Nisou s.r.o.**

Autor: Michal Seidl