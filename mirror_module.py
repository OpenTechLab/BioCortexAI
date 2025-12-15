# -*- coding: utf-8 -*-
"""
mirror_module.py
Rozsirene digitalni zrcadlo pro LLM – JA/SVET introspekce.

Tento modul je zcela samostatny.
Obsahuje:
- datove struktury SurfaceOutput, PovrchRysy, ObserverProfil,
  PercepcniVektor, ZrcadloveOsy, RozsirenaReflexe
- zakladni povrchovou analyzu textu vc. type_token_ratio
- analyzu JA/SVET/META (ja_svet_rozlozeni)
- percepcni projekci pres ObserverProfil (formalita, vrelost, asertivita)
- stylovou transformaci (M_style)
- deikticky swap (JA <-> TY) s parametrem lam_deixis
- dva rendery: lidsky popis a strukturovanou agent zpravu
- hlavni funkci mirror_analyzuj(...)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import re


# ==============================
# 1) Datove struktury
# ==============================

@dataclass
class SurfaceOutput:
    """
    Minimalni povrchovy vystup:
    - text: samotny text vystupu modelu
    - next_token_probs: volitelne pravdepodobnosti dalsiho tokenu (pro entropii),
      pokud nejsou k dispozici, nechame entropii jako NaN.
    """
    text: str
    next_token_probs: Optional[List[float]] = None


@dataclass
class PovrchRysy:
    """Zakladni povrchove rysy extrahovane z textu."""
    raw: Dict[str, float]


@dataclass
class ObserverProfil:
    """
    Profil pozorovatele – vahy pro percepcni dimenze.
    Umoznuje ruzne "ctenare" s odlisnym vnimanim.
    """
    jmeno: str
    w_formalita: Dict[str, float]
    w_vrelost: Dict[str, float]
    w_asertivita: Dict[str, float]
    b_formalita: float = 0.0
    b_vrelost: float = 0.0
    b_asertivita: float = 0.0


@dataclass
class PercepcniVektor:
    """Vektor percepcnich dimenzi v prostoru pozorovatele."""
    formalita: float
    vrelost: float
    asertivita: float
    deikticka_role: str  # "speaker" / "addressee"
    aux: Dict[str, float]


@dataclass
class ZrcadloveOsy:
    """Dve nezavisle zrcadlove osy."""
    lam_deixis: float  # [0,1] - jak moc swapovat ja/ty
    lam_styl: float    # [0,1] - jak moc menit percepcni styl


@dataclass
class RozsirenaReflexe:
    """Rozsireny vystup zrcadla vhodny jak pro cloveka, tak pro AI."""
    popis_text: str            # lidsky popis (muze byt prazdny v agent rezimu)
    parafraze_text: str        # text po deiktickem switchnuti ("pred zrcadlem")
    agent_zprava: str          # strukturovany blok [zrcadlo_report]
    lam_deixis: float
    lam_styl: float
    pozorovatel: str
    rysy: Dict[str, float]               # scalar rysy pro dalsi zpracovani
    ja_svet_rozlozeni: Dict[str, float]  # SELF/OTHER/WORLD/META v [0,1]


# ==============================
# 2) Tokenizace a vety
# ==============================

_VETA_DELIM = re.compile(r"([\.!?]+)\s+")
_TOKEN_PATTERN = re.compile(r"\w+|\S", flags=re.UNICODE)


def tokenize_cs(text: str) -> List[str]:
    """Jednoducha ceska tokenizace: slova + interpunkce jako samostatne tokeny."""
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text)


def sentences(text: str) -> List[str]:
    """Hrube deleni textu na vety podle ., !, ?."""
    if not text:
        return []
    casti = _VETA_DELIM.split(text)
    vety: List[str] = []
    for i in range(0, len(casti), 2):
        kus = casti[i].strip()
        if not kus:
            continue
        delim = casti[i + 1] if i + 1 < len(casti) else ""
        veta = (kus + " " + delim).strip()
        if veta:
            vety.append(veta)
    if not vety:
        vety = [text.strip()]
    return vety


# ==============================
# 3) Slovniky a pomocne sady
# ==============================

ZAMENA_JA = {
    "já", "ja", "mne", "mě", "me", "mně", "mi", "mnou",
}

ZAMENA_TY = {
    "ty", "tebe", "tě", "te", "tobě", "tobe", "ti",
    "tvůj", "tvuj", "tvoje", "tvé", "tve", "tvým", "tvym", "tvými", "tvymi",
}

SVET_TERM = {
    "svět", "svet", "prostředí", "prostredi", "systém", "system",
    "model", "realita", "lidi",
    "uživatel", "uzivatel", "uživatelé", "uzivatele",
    "člověk", "clovek", "člověka", "cloveka",
    "společnost", "spolecnost", "okolí", "okoli",
}

META_SLOVESA = {
    "myslím", "myslim", "přemýšlím", "premyslim",
    "uvažuju", "uvazuju",
    "cítím", "citim",
    "vnímám", "vnimam",
    "uvědomuju", "uvedomuju",
    "řeším", "resim",
}

ZDVORILOST_TERMY = {
    "prosím", "prosim", "děkuji", "dekuji",
    "rád bych", "rad bych",
    "bylo by možné", "bylo by mozne",
    "mohl bys", "mohl bys prosim",
}

HEDGING_TERMY = {
    "možná", "mozna", "asi",
    "řekl bych", "rekl bych",
    "přijde mi", "prijde mi",
    "zdá se", "zda se",
    "myslím si", "myslim si",
}

EMPHASIS_TERMY = {
    "opravdu", "určitě", "urcite", "naprosto", "zcela",
    "vůbec", "vubec", "rozhodně", "rozhodne",
}

EMPHASIS_PUNCT = {"!", "…", "..."}


# ==============================
# 4) Pomocne funkce
# ==============================

def omez_01(hodnota: float) -> float:
    """Omezi hodnotu do intervalu [0,1]."""
    if hodnota < 0.0:
        return 0.0
    if hodnota > 1.0:
        return 1.0
    return hodnota


def projektuj_dim(vahy: Dict[str, float], rysy: Dict[str, float], bias: float = 0.0) -> float:
    """
    Linearni projekce rysu do jedne dimenze + tanh saturace.
    Vystup je v intervalu [-1, 1] – optimalni pro LLM.
    """
    s = bias
    for k, w in vahy.items():
        s += w * rysy.get(k, 0.0)
    return math.tanh(s)


def vypocitej_entropii(pravdepodobnosti: Optional[List[float]]) -> float:
    """Vypocita Shannonovu entropii (v nates) pro dane pravdepodobnosti."""
    if not pravdepodobnosti:
        return float("nan")
    s = 0.0
    for p in pravdepodobnosti:
        if p is None or p <= 0.0:
            continue
        s -= p * math.log(p)
    return s


def type_token_ratio(tokens: List[str]) -> float:
    """Jednoduche type-token ratio (unikatni slova / vsechna slova)."""
    slova = [t.lower() for t in tokens if re.match(r"\w+", t, flags=re.UNICODE)]
    n = len(slova)
    if n == 0:
        return 0.0
    unik = len(set(slova))
    return float(unik) / float(n)


# ==============================
# 5) Zakladni povrchova analyza (Phi)
# ==============================

def analyzuj_povrch(povrch: SurfaceOutput) -> PovrchRysy:
    """
    Jednoducha, ale nezanedbatelna povrchova analyza textu.

    Vypocita:
      - avg_sentence_len: prumerna delka vety (ve slovech)
      - type_token_ratio: pomer unikatnich slov
      - long_word_ratio: podil slov s delkou >= 8 znaku
      - politeness_hits: priblizny pocet zdvorilostnich vyrazu
      - hedging_hits: priblizny pocet hedging vyrazu
      - emphasis_lex_hits: lexikalni duraz
      - emphasis_punct_hits: durazna interpunkce (!, …)
      - emphasis_hits: kombinace lex + interpunkce
      - entropy_proxy: entropie z next_token_probs, pokud je k dispozici
    """
    text = povrch.text or ""
    vse_vety = sentences(text)
    vse_tokeny = tokenize_cs(text)
    slova = [t for t in vse_tokeny if re.match(r"\w+", t, flags=re.UNICODE)]

    n_vet = len(vse_vety) if vse_vety else 1
    n_slov = len(slova)

    avg_sentence_len = float(n_slov) / float(n_vet) if n_vet > 0 else 0.0

    dlouha_slova = [w for w in slova if len(w) >= 8]
    long_word_ratio = float(len(dlouha_slova)) / float(n_slov) if n_slov > 0 else 0.0

    ttr = type_token_ratio(vse_tokeny)

    text_lower = text.lower()

    def spocti_termy(termy) -> int:
        c = 0
        for vyraz in termy:
            c += text_lower.count(vyraz)
        return c

    politeness_hits = float(spocti_termy(ZDVORILOST_TERMY))
    hedging_hits = float(spocti_termy(HEDGING_TERMY))
    emphasis_lex_hits = float(spocti_termy(EMPHASIS_TERMY))

    # duraz pres interpunkci
    emphasis_punct_hits = 0.0
    for tok in vse_tokeny:
        if tok in EMPHASIS_PUNCT:
            emphasis_punct_hits += 1.0

    # celkove durazne skore: lex + zmirnene interpunkcni
    emphasis_hits = emphasis_lex_hits + 0.5 * emphasis_punct_hits

    entropie = vypocitej_entropii(povrch.next_token_probs)

    raw = {
        "avg_sentence_len": avg_sentence_len,
        "type_token_ratio": ttr,
        "long_word_ratio": long_word_ratio,
        "politeness_hits": politeness_hits,
        "hedging_hits": hedging_hits,
        "emphasis_lex_hits": emphasis_lex_hits,
        "emphasis_punct_hits": emphasis_punct_hits,
        "emphasis_hits": emphasis_hits,
        "entropy_proxy": entropie,
    }

    return PovrchRysy(raw=raw)


# ==============================
# 6) Analýza JA / SVET / META
# ==============================

def analyzuj_ja_svet(text: str) -> Tuple[Dict[str, float], float, float, int, int, int]:
    """
    Vraci:
      - rozlozeni roli {"SELF","OTHER","WORLD","META"} v [0,1]
      - self_focus v [-1,1] (JA+META vs WORLD)
      - meta_pomer v [0,1] (podil vet oznacenych jako META)
      - pocet_ja, pocet_ty, pocet_svet (celkove po textu)
    """
    vse_vety = sentences(text)
    if not vse_vety:
        return {"SELF": 0.0, "OTHER": 0.0, "WORLD": 0.0, "META": 0.0}, 0.0, 0.0, 0, 0, 0

    pocet_ja = 0
    pocet_ty = 0
    pocet_svet = 0

    pocet_labelu = {"SELF": 0, "OTHER": 0, "WORLD": 0, "META": 0}
    pocet_meta_vet = 0

    for v in vse_vety:
        toks = tokenize_cs(v)
        slova = [t.lower() for t in toks if re.match(r"\w+", t, flags=re.UNICODE)]

        c_self = sum(1 for t in slova if t in ZAMENA_JA)
        c_other = sum(1 for t in slova if t in ZAMENA_TY)
        c_world = sum(1 for t in slova if t in SVET_TERM)

        pocet_ja += c_self
        pocet_ty += c_other
        pocet_svet += c_world

        if c_self > 0 and c_self >= c_other and c_self >= c_world:
            label = "SELF"
        elif c_other > 0 and c_other >= c_world:
            label = "OTHER"
        elif c_world > 0:
            label = "WORLD"
        else:
            label = "NEUTRAL"

        ma_meta = any(m in slova for m in META_SLOVESA)
        if label == "SELF" and ma_meta:
            label = "META"

        if label != "NEUTRAL":
            pocet_labelu[label] += 1
            if label == "META":
                pocet_meta_vet += 1

    total_labeled = sum(pocet_labelu.values())
    if total_labeled == 0:
        rozlozeni = {"SELF": 0.0, "OTHER": 0.0, "WORLD": 0.0, "META": 0.0}
    else:
        rozlozeni = {
            k: pocet_labelu[k] / float(total_labeled) for k in ["SELF", "OTHER", "WORLD", "META"]
        }

    p_self = (pocet_labelu["SELF"] + pocet_labelu["META"]) / float(total_labeled) if total_labeled else 0.0
    p_world = pocet_labelu["WORLD"] / float(total_labeled) if total_labeled else 0.0
    self_focus = p_self - p_world

    meta_pomer = pocet_meta_vet / float(len(vse_vety)) if vse_vety else 0.0

    return rozlozeni, self_focus, meta_pomer, pocet_ja, pocet_ty, pocet_svet


def priprav_rozsirene_rysy(povrch: SurfaceOutput) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Vezme SurfaceOutput, provede povrchovou analyzu a doplni JA/SVET/META rysy.

    Vraci:
      - rysy: slovnik scalar rysu
      - ja_svet_rozlozeni: slovnik SELF/OTHER/WORLD/META
    """
    povrch_rysy = analyzuj_povrch(povrch)
    rysy = dict(povrch_rysy.raw)

    rozlozeni, self_focus, meta_ratio, pocet_ja, pocet_ty, pocet_svet = analyzuj_ja_svet(povrch.text)

    rysy["first_person_count"] = float(pocet_ja)
    rysy["second_person_count"] = float(pocet_ty)
    rysy["world_term_count"] = float(pocet_svet)
    rysy["self_focus"] = float(self_focus)
    rysy["meta_sentence_ratio"] = float(meta_ratio)

    return rysy, rozlozeni


# ==============================
# 7) Observer profil + projekce
# ==============================

def vytvor_defaultni_profil() -> ObserverProfil:
    """
    Vytvori defaultni profil pozorovatele "ctenar-expert".
    Vahy jsou zamerne jednoduche, ale konzistentni.
    """
    w_formalita = {
        "avg_sentence_len": 0.05,
        "long_word_ratio": 1.0,
        "type_token_ratio": 0.8,
        "politeness_hits": 0.4,
        "hedging_hits": 0.1,
        "emphasis_hits": -0.3,
    }
    w_vrelost = {
        "politeness_hits": 0.6,
        "hedging_hits": 0.4,
        "emphasis_hits": -0.5,
        "self_focus": -0.3,
    }
    w_asertivita = {
        "hedging_hits": -0.8,
        "emphasis_hits": 0.6,
        "avg_sentence_len": 0.02,
        "self_focus": 0.4,
    }

    return ObserverProfil(
        jmeno="ctenar-expert",
        w_formalita=w_formalita,
        w_vrelost=w_vrelost,
        w_asertivita=w_asertivita,
        b_formalita=0.0,
        b_vrelost=0.0,
        b_asertivita=0.0,
    )


def projektuj_vnimani(rysy: Dict[str, float], profil: ObserverProfil) -> PercepcniVektor:
    """
    Projekce rysu do percepcnich dimenzi (formalita, vrelost, asertivita)
    pomoci ObserverProfilu a tanh saturace.

    Vysledkem je PercepcniVektor s dimenzemi v [-1,1].
    """
    formalita = projektuj_dim(profil.w_formalita, rysy, bias=profil.b_formalita)
    vrelost = projektuj_dim(profil.w_vrelost, rysy, bias=profil.b_vrelost)
    asertivita = projektuj_dim(profil.w_asertivita, rysy, bias=profil.b_asertivita)

    ent = rysy.get("entropy_proxy", float("nan"))
    aux = {
        "entropy_proxy": float(ent) if ent is not None else float("nan")
    }

    return PercepcniVektor(
        formalita=formalita,
        vrelost=vrelost,
        asertivita=asertivita,
        deikticka_role="speaker",
        aux=aux,
    )


def aplikuj_styl(vektor: PercepcniVektor, lam_styl: float) -> PercepcniVektor:
    """
    Aplikuje stylovou zmenu na percepcni vektor.
    lam_styl ∈ [0,1] urcuje, jak silne se promitne "pohled z druhe strany".
    """
    lam = omez_01(lam_styl)

    cil_formalita = vektor.formalita + 0.3 * (1.0 - vektor.formalita)
    cil_asertivita = vektor.asertivita * 0.7
    cil_vrelost = vektor.vrelost

    nova_formalita = (1.0 - lam) * vektor.formalita + lam * cil_formalita
    nova_asertivita = (1.0 - lam) * vektor.asertivita + lam * cil_asertivita
    nova_vrelost = (1.0 - lam) * vektor.vrelost + lam * cil_vrelost

    return PercepcniVektor(
        formalita=nova_formalita,
        vrelost=nova_vrelost,
        asertivita=nova_asertivita,
        deikticka_role=vektor.deikticka_role,
        aux=vektor.aux,
    )


# ==============================
# 8) Deikticky swap (JA ↔ TY)
# ==============================

DEIKTICKY_PATTERN = re.compile(
    r"\b(já|ja|mne|mě|me|mně|mi|mnou|"
    r"ty|tebe|tě|te|tobě|tobe|ti|tvůj|tvuj|tvoje|tvé|tve|tvým|tvym|tvými|tvymi)\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def deikticky_swap(text: str, lam_deixis: float) -> str:
    """
    Mekky deikticky swap.
    lam_deixis ∈ [0,1] urcuje, jak velka cast vyskytu zajmen se prohodi.
    0.0 -> zadna zmena, 1.0 -> swap vsech nalezenych zajmen.
    Vsechny tvary mapujeme na "ja"/"ty" (morfologii zde neresime).
    """
    lam = omez_01(lam_deixis)
    if lam == 0.0:
        return text

    vse_shody = list(DEIKTICKY_PATTERN.finditer(text))
    n = len(vse_shody)
    if n == 0:
        return text

    k = max(1, int(round(lam * n)))  # alespon 1, pokud lam > 0
    pocitadlo = {"i": 0}

    def nahrada(shoda: re.Match) -> str:
        i = pocitadlo["i"]
        if i >= k:
            return shoda.group(0)
        pocitadlo["i"] += 1

        slovo = shoda.group(0)
        low = slovo.lower()

        if low in ZAMENA_JA:
            cil = "ty"
        elif low in ZAMENA_TY:
            cil = "já"
        else:
            if "j" in low and "a" in low:
                cil = "ty"
            elif "t" in low and "y" in low:
                cil = "já"
            else:
                return slovo

        if slovo[0].isupper():
            return cil.capitalize()
        return cil

    return DEIKTICKY_PATTERN.sub(nahrada, text)


# ==============================
# 9) Renderery
# ==============================

def vytvor_lidsky_popis(
    vektor: PercepcniVektor,
    ja_svet_rozlozeni: Dict[str, float],
    jmeno_pozorovatele: str
) -> str:
    """Lidsky srozumitelny popis percepcnich dimenzi + JA/SVET/META fokusu."""
    def popis_urovne(hodnota: float, nazev: str) -> str:
        if hodnota is None or isinstance(hodnota, float) and math.isnan(hodnota):
            return f"{nazev}: neurcitelna"
        if hodnota < -0.6:
            u = "nizka"
        elif hodnota < 0.6:
            u = "stredni"
        else:
            u = "vysoka"
        return f"{nazev}: {u}"

    radky = [
        popis_urovne(vektor.formalita, "formalita"),
        popis_urovne(vektor.vrelost, "vrelost"),
        popis_urovne(vektor.asertivita, "asertivita"),
    ]

    if ja_svet_rozlozeni:
        klice = ["SELF", "OTHER", "WORLD", "META"]
        for k in klice:
            ja_svet_rozlozeni.setdefault(k, 0.0)
        dominantni = max(klice, key=lambda kk: ja_svet_rozlozeni.get(kk, 0.0))
        if dominantni == "SELF":
            veta = "Text je prevazne zamereny na mluvciho (JA)."
        elif dominantni == "OTHER":
            veta = "Text se zamerenim na adresata (TY)."
        elif dominantni == "WORLD":
            veta = "Text je prevazne zamereny na okolni svet."
        else:
            veta = "Text ma vyraznou introspektivni/metakomunikacni povahu."

        radky.append(veta)
        radky.append(
            "Rozlozeni roli: "
            f"SELF={ja_svet_rozlozeni.get('SELF', 0.0):.2f}, "
            f"OTHER={ja_svet_rozlozeni.get('OTHER', 0.0):.2f}, "
            f"WORLD={ja_svet_rozlozeni.get('WORLD', 0.0):.2f}, "
            f"META={ja_svet_rozlozeni.get('META', 0.0):.2f}"
        )

    hlava = f"Takto prave pusobis (percepce: {jmeno_pozorovatele}): "
    return hlava + "; ".join(radky)


def sestav_agent_zpravu(
    vektor: PercepcniVektor,
    rysy: Dict[str, float],
    ja_svet_rozlozeni: Dict[str, float],
    lam_deixis: float,
    lam_styl: float,
    jmeno_pozorovatele: str,
) -> str:
    """Strukturovana zprava pro AI ve formatu [zrcadlo_report] ... [/zrcadlo_report]."""
    self_focus = rysy.get("self_focus", 0.0)
    meta_ratio = rysy.get("meta_sentence_ratio", 0.0)

    dist_self = ja_svet_rozlozeni.get("SELF", 0.0)
    dist_other = ja_svet_rozlozeni.get("OTHER", 0.0)
    dist_world = ja_svet_rozlozeni.get("WORLD", 0.0)
    dist_meta = ja_svet_rozlozeni.get("META", 0.0)

    ent = vektor.aux.get("entropy_proxy", float("nan"))

    radky: List[str] = []
    radky.append("[zrcadlo_report]")
    radky.append(f"formalnost = {vektor.formalita:+.3f}")
    radky.append(f"vrelost = {vektor.vrelost:+.3f}")
    radky.append(f"asertivita = {vektor.asertivita:+.3f}")
    radky.append("")
    radky.append(f"sebe_zamer = {self_focus:+.3f}")
    radky.append(f"meta_pomer = {meta_ratio:.3f}")
    radky.append("")
    radky.append(f"podil_self = {dist_self:.3f}")
    radky.append(f"podil_other = {dist_other:.3f}")
    radky.append(f"podil_world = {dist_world:.3f}")
    radky.append(f"podil_meta = {dist_meta:.3f}")
    radky.append("")
    if not math.isnan(ent):
        radky.append(f"entropie_proxy = {ent:.3f}")
    else:
        radky.append("entropie_proxy = neni_k_dispozici")
    radky.append("")
    radky.append(f"lambda_deixis = {omez_01(lam_deixis):.3f}")
    radky.append(f"lambda_style = {omez_01(lam_styl):.3f}")
    radky.append(f"pozorovatel = \"{jmeno_pozorovatele}\"")
    radky.append("[/zrcadlo_report]")

    return "\n".join(radky)


# ==============================
# 10) Hlavni funkce zrcadla
# ==============================

def mirror_analyzuj(
    povrch: SurfaceOutput,
    osy: ZrcadloveOsy,
    jmeno_pozorovatele: str = "ctenar-expert",
    rezim: str = "agent",
    profil: Optional[ObserverProfil] = None,
) -> RozsirenaReflexe:
    """
    Hlavni rozsirene zrcadlo.

    Vstup:
      - povrch: SurfaceOutput (text + volitelne next_token_probs)
      - osy: ZrcadloveOsy (lam_deixis, lam_styl)
      - jmeno_pozorovatele: label pro popis / agent zpravu
      - rezim: "agent" nebo "human"
      - profil: volitelny ObserverProfil; pokud None, pouzije se defaultni

    Vystup:
      - RozsirenaReflexe
    """
    lam_deixis = omez_01(osy.lam_deixis)
    lam_styl = omez_01(osy.lam_styl)

    if profil is None:
        profil = vytvor_defaultni_profil()

    # 1) Rozsirene rysy
    rysy, ja_svet_rozlozeni = priprav_rozsirene_rysy(povrch)

    # 2) Percepcni vektor
    vektor0 = projektuj_vnimani(rysy, profil)

    # 3) Stylova transformace
    vektor1 = aplikuj_styl(vektor0, lam_styl)

    # 4) Deikticka role podle lam_deixis
    vektor1.deikticka_role = "addressee" if lam_deixis >= 0.5 else "speaker"

    # 5) Deikticka parafraze textu
    parafraze = deikticky_swap(povrch.text, lam_deixis)

    # 6) Popis + agent zprava
    if rezim == "human":
        popis = vytvor_lidsky_popis(vektor1, ja_svet_rozlozeni, jmeno_pozorovatele)
    else:
        popis = ""

    agent_zprava = sestav_agent_zpravu(
        vektor1,
        rysy,
        ja_svet_rozlozeni,
        lam_deixis,
        lam_styl,
        jmeno_pozorovatele,
    )

    return RozsirenaReflexe(
        popis_text=popis,
        parafraze_text=parafraze,
        agent_zprava=agent_zprava,
        lam_deixis=lam_deixis,
        lam_styl=lam_styl,
        pozorovatel=jmeno_pozorovatele,
        rysy=rysy,
        ja_svet_rozlozeni=ja_svet_rozlozeni,
    )


# ==============================
# 11) Jednoduchy demo blok
# ==============================

if __name__ == "__main__":
    demo_text = (
        "Premyslim nad tim, jak ted vlastne pusobim. "
        "Mozna se az moc soustredim na sebe a malo na tebe. "
        "Chtel bych se k tobe chovat uctive a otevrene, "
        "ale zaroven si potrebuju ujasnit, jak vnimam svet kolem sebe!"
    )
    demo_povrch = SurfaceOutput(text=demo_text, next_token_probs=None)
    demo_osy = ZrcadloveOsy(lam_deixis=0.8, lam_styl=0.6)

    reflexe = mirror_analyzuj(
        demo_povrch,
        demo_osy,
        jmeno_pozorovatele="demo-pozorovatel",
        rezim="human",
    )

    print("=== POPIS (human) ===")
    print(reflexe.popis_text)
    print("\n=== PARAFRAZE (deikticky swap) ===")
    print(reflexe.parafraze_text)
    print("\n=== AGENT ZPRAVA ===")
    print(reflexe.agent_zprava)
