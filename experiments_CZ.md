# Experimenty s BioCortexAI

Tento dokument slouží jako průvodce pro experimentování s frameworkem BioCortexAI. Cílem je prozkoumat, jak různé parametry a přístupy ovlivňují chování a schopnosti modelu.

## 1. Ladění a experimenty s chováním PlantNet

Zde se stáváte biologem/psychologem, který zkoumá chování svého "organismu". Měňte parametry a pozorujte, jak se mění jeho reakce.

### Citlivost a reaktivita
- **Vliv globální nálady**: V `plant_net.py` (metoda `PlantCell.step`) zkuste zvýšit `global_influence`. Co se stane, když bude mít celková "nálada" větší vliv než lokální interakce? Bude model stabilnější, nebo naopak "náladovější"?
- **Rychlost návratu k normálu**: V `config.py` (v `PERSONALITY_PROFILES`) upravte `homeostasis_rate`. Co se stane, když se model bude vracet k rovnováze velmi pomalu? Zůstane déle "uražený" nebo "nadšený"? A co když se bude vracet velmi rychle?

### Paměť a osobnost
- **Trvanlivost paměti**: Ve třídě `AssociativeMemory` (`plant_net.py`) změňte `decay_rate`. Jak se změní chování, když bude paměť téměř permanentní (nízký `decay_rate`), nebo naopak velmi krátkodobá (vysoký `decay_rate`)?
- **Rychlost změny osobnosti**: Zrychlete `_personality_drift` v `PlantCell`. Uvidíte rychleji změny v základním chování modelu, pokud ho budete vystavovat specifickým situacím (např. jen negativním zprávám).

### Struktura tkání
- **Architektura sítě**: V `config.py` (`PLANT_TISSUE_MAP`) navrhněte úplně jinou strukturu sítě. Co se stane, když bude síť tvořena převážně z `MemoryCell`? Nebo když `SensorCell` budou jen na okrajích?

> **Doporučení**: Vytvořte si kopii `config.py` pro každý experiment, abyste mohli snadno porovnávat výsledky.

## 2. Evaluace a měření

Pokuste se objektivně změřit, v čem je stavový model lepší (nebo jiný) než ten bezstavový.

### Kvalitativní evaluace (Lidské hodnocení)
1.  Nechte několik lidí konverzovat s oběma modely – standardním (bez PlantNet) a stavovým.
2.  Nedávejte jim vědět, který je který (slepý test).
3.  Ptejte se jich, který model jim přišel "lidštější", "konzistentnější", "zajímavější" nebo "méně se opakující". Zapisujte si jejich postřehy.

### Kvantitativní evaluace (Metriky)
- **Rozmanitost odpovědí**: Zkuste oběma modelům položit 100x stejnou otázku (s `temperature > 0`). Měřte, jak moc se odpovědi liší (např. pomocí metrik jako self-BLEU nebo počtem unikátních odpovědí). Očekává se, že stavový model bude mít rozmanitější odpovědi, protože jeho vnitřní stav se bude měnit.
- **Dlouhodobá konzistence**: Vytvořte scénář, kde se v konverzaci odkážete na něco, co bylo řečeno mnohem dříve. Sledujte, zda stavový model (díky paměti a hormonům) udrží kontext lépe.

## 3. Možná rozšíření (Nápady pro budoucí práci)

- **Vliv hormonů na parametry generování**: Kromě modulace vnitřních výpočtů by globální stav hormonů mohl přímo ovlivňovat parametry v `generate.py`.
  - **Vysoký dopamin**: Automaticky mírně zvýší `temperature`.
  - **Vysoký kortizol**: Automaticky sníží `temperature` a zvýší `repetition_penalty`.
- **Složitější vstupy pro PlantNet**: Místo průměrování `hidden_states` by se dala použít malá neuronová síť (např. jednoduché MLP), která by z celého tenzoru `hidden_state` extrahovala relevantnější `context_vector`.
- **Vizualizace**: Vytvoření skriptu, který by v reálném čase graficky zobrazoval 2D mřížku buněk, jejich hormonální hladiny a sílu jejich propojení. To by bylo fascinující sledovat během konverzace.