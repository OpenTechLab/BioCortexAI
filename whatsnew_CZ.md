======================================================================
## OpenTechLab Jablonec nad Nisou s. r. o.
### (c)2025
# CO JE NOVÉHO: VERZE ALFA 2.0 vs. VERZE BETA 1.0 (Hormonální sítě)
======================================================================

Tento dokument popisuje klíčové koncepční a architektonické změny mezi experimentální verzí Alfa 2.0 a současnou verzí Beta 1.0 frameworku BioCortexAI.

---

## A. Zásadní koncepční a architektonické rozdíly

Toto jsou největší změny v myšlení a designu celého systému.

### 1. Zdroj "inteligence"
- **VERZE ALFA 2.0**: Jedna neuronová síť (`PlantHormoneNetwork`) se učí aproximovat "správné" hormonální reakce pomocí zpětné propagace. Jde o **centralizované učení**.
- **VERZE BETA 1.0**: Inteligentní chování je **emergentním jevem** z distribuované sítě buněk. Chování není naučeno jednou sítí, ale vyplývá z lokálních interakcí, pravidel a samoorganizace.
- ***POSUN***: Od centralizovaného učení k **decentralizované, samoorganizující se inteligenci**. Verze Beta je mnohem blíže původní biologické inspiraci.

### 2. Mechanismus učení
- **VERZE ALFA 2.0**: Zpětná propagace a optimalizátor (`Adam`). Síť minimalizuje explicitně definovanou `loss` funkci.
- **VERZE BETA 1.0**: Adaptace v reálném čase (plasticita). Neexistuje `loss` funkce. Síť se adaptuje přímo na základě interakcí (paměť, drift osobnosti, změna konektivity).
- ***POSUN***: Od explicitního tréninku k **implicitní, kontinuální adaptaci**. Model se neučí "správnou odpověď", ale neustále se přizpůsobuje svému prostředí.

### 3. Struktura sítě
- **VERZE ALFA 2.0**: Monolitická, centralizovaná. Jeden objekt `PlantHormoneNetwork` řídí vše.
- **VERZE BETA 1.0**: Distribuovaná 2D mřížka (`DistributedPlantNetwork`). Každá buňka má vlastní stav a paměť.
- ***POSUN***: Od "černé skříňky" k **interpretovatelnému, modulárnímu systému**, kde lze definovat tkáně a specializované buňky.

### 4. "Osobnost"
- **VERZE ALFA 2.0**: Neexistuje. Chování je plně dáno naučenými váhami.
- **VERZE BETA 1.0**: Explicitně definovaná a dynamická. Každá buňka má osobnostní profil (`OPTIMISTIC`, `CURIOUS`...), který ovlivňuje její reakce a může se pomalu měnit (`_personality_drift`).
- ***POSUN***: Model získává **stabilní, ale plastickou povahu**. Jeho základní chování je předvídatelné, ale může se dlouhodobě vyvíjet.

### 5. Paměť
- **VERZE ALFA 2.0**: Implicitní. Paměť je zakódována ve vahách `PlantHormoneNetwork` a jako jednoduchý `memory_feedback` vektor.
- **VERZE BETA 1.0**: Explicitní, asociativní a s **významovým zapomínáním** (`AssociativeMemory`). Model si ukládá konkrétní "vzpomínky" a emočně nabité zážitky přetrvávají déle.
- ***POSUN***: Model získává schopnost pamatovat si konkrétní události, což je mnohem silnější forma paměti.

### 6. Integrace s LLM
- **VERZE ALFA 2.0**: Ad-hoc a invazivní. Modulace se děje hluboko uvnitř `ModulatedMultiHeadAttention`.
- **VERZE BETA 1.0**: Čistá, neinvazivní a oddělená. Modulace se děje na úrovni `TransformerBlock` přes čisté API (`PlantLayer`). Trénink LLM je zcela izolován.
- ***POSUN***: Verze Beta má **profesionální a robustní integraci**. LLM lze snadno vyměnit nebo používat bez sítě PlantNet.

---

## B. Konkrétní implementační rozdíly

| Aspekt | Verze Alfa 2.0 (`plant_hormone_network.py`) | Verze Beta 1.0 (`plant_net.py`) |
| :--- | :--- | :--- |
| **Jádro logiky** | Jedna třída `PlantHormoneNetwork(nn.Module)`. | Ekosystém tříd: `PlantLayer`, `DistributedPlantNetwork`, `PlantCell`, `AssociativeMemory`. |
| **Výpočet hormonů** | `forward` pass neuronové sítě. | Deterministicky na základě pravidel. |
| **Vstupy** | Ručně vytvořené tenzory s pevnou dimenzí. | Přirozené výstupy z LLM (skaláry, vektory). |
| **Uložení stavu** | `state_dict` neuronové sítě. | Kompletní snapshot sítě do JSON. |
| **Architektura LLM** | Klasický Transformer (Encoder-Only). | Moderní Transformer (Decoder-Only). |
| **Poziční kódování**| Standardní `PositionalEncoding`. | RoPE (Rotary Positional Embeddings). |
| **Technologie LLM** | Standardní `Attention`. | SwiGLU, Grouped-Query Attention (GQA). |
| **Mechanismus modulace** | Komplexní modifikace `attn_scores`. | Čisté násobení `Value` vektorů a reziduálních spojení. |
| **Trénink** | Trénuje se LLM i hormonální síť. | Trénuje se **POUZE** LLM. Rostlinná síť se pouze adaptuje. |

---

## C. Shrnutí: Evoluce myšlenky

- **VERZE ALFA 2.0**:
  Byla úspěšným **důkazem konceptu (Proof of Concept)**. Ukázala, že myšlenka modulace LLM pomocí externího signálu je možná. Použila k tomu standardní nástroje (jedna NN, zpětná propagace) a integrovala je přímo a natvrdo.

- **VERZE BETA 1.0**:
  Je **zralá a promyšlená architektura**. Opouští myšlenku jedné "chytré" sítě a místo toho staví systém, jehož inteligence a adaptabilita **EMERGUJE** z interakce mnoha jednoduchých komponent. Tento systém je:
  - Blíže biologické realitě.
  - Robustnější a flexibilnější.
  - Lépe interpretovatelný.
  - Mnohem čistěji integrovaný do moderní LLM architektury.