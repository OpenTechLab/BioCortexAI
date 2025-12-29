# plant_net.py
import numpy as np
from collections import deque
import time
import json
import os
import torch

import config
from config import Personality
from sentiment_analyzer import sentiment_analyzer

PERSONALITY_PROFILES = {
    Personality.OPTIMISTIC: {"homeostasis_rate": 0.06, "sensitivities": {"entropy": 0.8, "latent_dev": 1.2, "sentiment": 1.2, "memory": 1.1}},
    Personality.CAUTIOUS:   {"homeostasis_rate": 0.10, "sensitivities": {"entropy": 1.2, "latent_dev": 0.8, "sentiment": 1.5, "memory": 1.1}},
    Personality.CURIOUS:    {"homeostasis_rate": 0.08, "sensitivities": {"entropy": 1.1, "latent_dev": 1.3, "sentiment": 1.0, "memory": 1.1}}
}

class AssociativeMemory:
    def __init__(self, max_size=200, decay_rate=0.0002):
        self.max_size, self.decay_rate, self.memory = max_size, decay_rate, deque()
    def _decay(self):
        new_memory, now = deque(), time.time()
        for embedding, reaction, hormones, timestamp, weight in self.memory:
            age = now - timestamp; significance = np.clip(max(abs(h - 1.0) for h in hormones.values()), 0.0, 1.0)
            new_weight = weight * np.exp(-self.decay_rate * (1 - significance) * age)
            if new_weight > 0.01: new_memory.append((embedding, reaction, hormones, timestamp, new_weight))
        self.memory = new_memory
    def add(self, embedding, reaction, hormones, weight=1.0):
        if len(self.memory) >= self.max_size:
            self._decay()
            if len(self.memory) >= self.max_size: self.memory.popleft()
        self.memory.append((embedding.tolist(), reaction, hormones, time.time(), weight))
    def retrieve_signal(self, query_embedding):
        if not self.memory: return 0.0
        self._decay()
        if not self.memory: return 0.0
        sims, q_norm = [], np.linalg.norm(query_embedding)
        for emb_list, _, _, timestamp, weight in self.memory:
            embedding = np.array(emb_list); denom = (q_norm * np.linalg.norm(embedding) + 1e-8)
            sim = np.dot(query_embedding, embedding) / denom if denom != 0 else 0.0
            age = time.time() - timestamp; rarity = (1 - sim) * (age / (age + 10)); sims.append(rarity * weight)
        return max(sims) if sims else 0.0
    def to_dict(self): return {"max_size": self.max_size, "decay_rate": self.decay_rate, "memory": list(self.memory)}
    @classmethod
    def from_dict(cls, data):
        mem = cls(max_size=data["max_size"], decay_rate=data["decay_rate"])
        mem.memory = deque([(np.array(e), r, h, t, w) for e, r, h, t, w in data["memory"]]); return mem
class PlantCell:
    HORMONE_RANGE = (0.5, 1.5)
    def __init__(self, personality=Personality.OPTIMISTIC):
        profile = PERSONALITY_PROFILES[personality]; self.personality_enum = personality
        self.hormones = {"dopamin": 1.0, "serotonin": 1.0, "kortizol": 1.0, "oxytocin": 1.0}
        self.equilibrium = self.hormones.copy(); self.homeostasis_rate = profile["homeostasis_rate"]
        self.input_sensitivity = profile["sensitivities"].copy(); self.memory = AssociativeMemory()
        self.connection_strengths, self.connectivity_drift_rate = {}, 0.01; self.cell_type = self.__class__.__name__
    def step(self, entropy, latent_dev, sentiment, context_vector, neighbor_signal, global_signal):
        sensitivity = self.input_sensitivity; memory_signal = self.memory.retrieve_signal(context_vector)
        entropie_effect = entropy * (self.hormones["kortizol"] * 0.5)
        delta = {
            "dopamin":  self.input_sensitivity["latent_dev"] * (0.015 * latent_dev) \
                      + self.input_sensitivity["memory"]     * (0.020 * memory_signal) \
                      + self.input_sensitivity["sentiment"]  * (0.020 * sentiment) \
                      - (0.010 * entropie_effect),
            "serotonin": self.input_sensitivity["sentiment"] * (0.015 * sentiment) \
                      - (0.008 * entropie_effect) - (0.008 * latent_dev),
            "kortizol":  self.input_sensitivity["entropy"]   * (0.004 * entropie_effect) \
                      - self.input_sensitivity["sentiment"]  * (0.030 * sentiment),
            "oxytocin":  self.input_sensitivity["memory"]    * (0.015 * memory_signal) \
                      + self.input_sensitivity["sentiment"]  * (0.010 * sentiment) \
                      + (0.005 * (1 - latent_dev))
        }

        inertia_factor = 0.05
        delta['serotonin'] += (self.hormones['serotonin'] - 1.0) * inertia_factor
        delta['dopamin'] += (self.hormones['dopamin'] - 1.0) * inertia_factor

        for h in self.hormones: self.hormones[h] += delta[h]
        local_influence, global_influence = 0.2, 0.05
        for h in self.hormones:
            self.hormones[h] += local_influence * (neighbor_signal.get(h, self.hormones[h]) - self.hormones[h])
            self.hormones[h] += global_influence * (global_signal.get(h, self.hormones[h]) - self.hormones[h])
        self._asymmetric_homeostasis(); self._personality_drift(); self._update_connectivity()
        for h in self.hormones: self.hormones[h] = np.clip(self.hormones[h], self.HORMONE_RANGE[0], self.HORMONE_RANGE[1])

        try:
            # 1) skóre novosti a rarity (čím vyšší, tím větší motivace zapisovat)
            novelty = 0.5 * float(latent_dev) + 0.5 * float(min(1.0, float(entropy) / 5.0))
            rarity = max(0.0, 1.0 - float(self.memory.retrieve_signal(context_vector)))

            # 2) jednoduché prahy (lze později doladit podle logů)
            should_write = (novelty > 0.20) and (rarity > 0.30)

            if should_write:
                reaction = {
                    "entropy":   float(entropy),
                    "latent_dev": float(latent_dev),
                    "sentiment": float(sentiment)
                }
                # Váhu paměťové stopy navážeme na novost i raritu (0..1)
                weight = float(np.clip(0.5 * novelty + 0.5 * rarity, 0.05, 1.0))

                self.memory.add(
                    embedding=np.asarray(context_vector, dtype=np.float32),
                    reaction=reaction,
                    hormones={k: float(v) for k, v in self.hormones.items()},
                    weight=weight
                )
        except Exception:
            # Fail-safe: paměť nesmí rozbít krok buňky
            pass

    def _asymmetric_homeostasis(self):
        for h in self.hormones:
            diff = self.equilibrium[h] - self.hormones[h]
            self.hormones[h] += self.homeostasis_rate * (1.5 if diff > 0 else 0.5) * diff
    def _personality_drift(self):
        for h in self.equilibrium: self.equilibrium[h] += 0.0001 * (self.hormones[h] - self.equilibrium[h])
    def _update_connectivity(self):
        for neighbor_pos, strength in self.connection_strengths.items():
            strength_mod = (self.hormones['oxytocin'] - 1.0) - (self.hormones['kortizol'] - 1.0)
            new_strength = strength + self.connectivity_drift_rate * strength_mod * (1-strength) * strength
            self.connection_strengths[neighbor_pos] = np.clip(new_strength, 0.1, 1.0)
    def to_dict(self): return {"cell_type": self.cell_type, "personality": self.personality_enum.name, "hormones": self.hormones, "equilibrium": self.equilibrium, "memory": self.memory.to_dict(), "connection_strengths": {str(k): v for k, v in self.connection_strengths.items()}}
    @classmethod
    def from_dict(cls, data):
        cell = globals().get(data["cell_type"], PlantCell)(Personality[data["personality"]])
        cell.hormones, cell.equilibrium = data["hormones"], data["equilibrium"]
        cell.memory = AssociativeMemory.from_dict(data["memory"])
        cell.connection_strengths = {eval(k): v for k, v in data.get("connection_strengths", {}).items()}; return cell
class SensorCell(PlantCell):
    def __init__(self, personality=Personality.CURIOUS): super().__init__(personality); self.input_sensitivity["entropy"]*=1.5; self.input_sensitivity["sentiment"]*=1.5
class MemoryCell(PlantCell):
    def __init__(self, personality=Personality.CAUTIOUS): super().__init__(personality); self.memory = AssociativeMemory(max_size=500, decay_rate=0.00015)
class StructuralCell(PlantCell):
    def __init__(self, personality=Personality.OPTIMISTIC):
        super().__init__(personality); self.connectivity_drift_rate=0.005
        for k in self.input_sensitivity: self.input_sensitivity[k]*=0.5
class DistributedPlantNetwork:
    def __init__(self, width, height, tissue_map):
        self.width, self.height = width, height
        self.cells = [[globals().get(name, PlantCell)(p) for name, p in row] for row in tissue_map]
        self._initialize_connectivity()

    def _get_neighbors(self, x, y):
        return [
            (nx, ny)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= (nx := x + dx) < self.width and 0 <= (ny := y + dy) < self.height
        ]

    def _initialize_connectivity(self):
        for y, r in enumerate(self.cells):
            for x, c in enumerate(r):
                for n_pos in self._get_neighbors(x, y):
                    c.connection_strengths[n_pos] = 0.8 if isinstance(c, StructuralCell) else 0.5

    def step(self, entropy, latent_dev, sentiment, context_vector):
        global_state = self.global_hormone_state()

        entropy_grid = np.full((self.height, self.width), entropy)
        latent_dev_grid = np.full((self.height, self.width), latent_dev)
        sentiment_grid = np.full((self.height, self.width), sentiment)
        context_vectors_grid = np.tile(context_vector, (self.height, self.width, 1))

        neighbor_hormones = {}
        neighbor_weights = {}
        for y, r in enumerate(self.cells):
            for x, c in enumerate(r):
                n_h = {h: [] for h in c.hormones}
                w_list = []
                for nx, ny in self._get_neighbors(x, y):
                    neighbor = self.cells[ny][nx]
                    weight = c.connection_strengths.get((nx, ny), 1.0)
                    w_list.append(weight)
                    for h in c.hormones:
                        n_h[h].append(neighbor.hormones[h] * weight)
                neighbor_hormones[(x, y)] = n_h
                neighbor_weights[(x, y)] = w_list

        for y, r in enumerate(self.cells):
            for x, c in enumerate(r):
                n_h = neighbor_hormones[(x, y)]
                w_list = neighbor_weights[(x, y)]
                avg_n_s = {
                    h: sum(n_h[h]) / (sum(w_list) if w_list else 1)
                    for h in c.hormones
                }
                c.step(
                    entropy_grid[y, x],
                    latent_dev_grid[y, x],
                    sentiment_grid[y, x],
                    context_vectors_grid[y, x],
                    avg_n_s,
                    global_state
                )

    def global_hormone_state(self):
        all_h = {h: [] for h in self.cells[0][0].hormones}
        for r in self.cells:
            for c in r:
                for h in c.hormones:
                    all_h[h].append(c.hormones[h])
        return {h: np.mean(v) for h, v in all_h.items()}

    def save(self, fpath):
        with open(fpath, "w", encoding='utf-8') as f:
            json.dump({
                "width": self.width,
                "height": self.height,
                "cells": [[c.to_dict() for c in r] for r in self.cells]
            }, f, indent=4)

    @classmethod
    def load(cls, fpath):
        with open(fpath, "r", encoding='utf-8') as f:
            state = json.load(f)
        net = cls(
            state["width"],
            state["height"],
            [[(c['cell_type'], Personality[c['personality']]) for c in r] for r in state['cells']]
        )
        net.cells = [[PlantCell.from_dict(c) for c in r] for r in state["cells"]]
        return net

    @classmethod
    def from_dict_static(cls, state):
        width, height = state["width"], state["height"]
        temp_tissue_map = [
            [(c['cell_type'], Personality[c['personality']]) for c in r]
            for r in state['cells']
        ]
        net = cls(width, height, temp_tissue_map)
        net.cells = [[PlantCell.from_dict(c) for c in row] for row in state["cells"]]
        return net


class PlantLayer:
    def __init__(self, state_file=config.PLANT_NET_STATE_FILE):
        self.tissue_map = config.PLANT_TISSUE_MAP
        self.height, self.width = len(self.tissue_map), len(self.tissue_map[0])
        self.last_hidden_state_avg = None
        self.state_file = state_file
        if state_file and os.path.exists(state_file):
            print(f"PlantLayer: Loading state form '{state_file}'")
            self.network = DistributedPlantNetwork.load(state_file)
        else:
            print("PlantLayer: I am creating a new network according to the tissue map from the config.")
            self.network = DistributedPlantNetwork(self.width, self.height, self.tissue_map)
        self._global_ema = None

    @torch.no_grad()
    def _calculate_inputs(self, logits, hidden_states, prompt_text: str):
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
        current_hs_avg = hidden_states[:, -1, :].mean(dim=0)
        if self.last_hidden_state_avg is None: self.last_hidden_state_avg = current_hs_avg
        latent_dev = 1.0 - torch.cosine_similarity(self.last_hidden_state_avg, current_hs_avg, dim=0).item()
        self.last_hidden_state_avg = 0.1 * current_hs_avg + 0.9 * self.last_hidden_state_avg
        
        sentiment = sentiment_analyzer.get_score(prompt_text)
        
        context_vector = hidden_states[:, -1, :].mean(dim=0).cpu().numpy()
        return entropy, latent_dev, sentiment, context_vector

    def update_state(self, logits, hidden_states, prompt_text: str):
        entropy, latent_dev, sentiment, context_vector = self._calculate_inputs(
            logits, hidden_states, prompt_text
        )
        self.network.step(entropy, latent_dev, sentiment, context_vector)
        return sentiment

    def get_global_hormones(self):
        raw = self.network.global_hormone_state()
        alpha = 0.10  
        if self._global_ema is None:
            self._global_ema = raw.copy()
        else:
            for h in raw:
                self._global_ema[h] = (1 - alpha) * self._global_ema[h] + alpha * raw[h]
        return self._global_ema

    def save_state(self):
        if self.state_file:
            self.network.save(self.state_file)
            print(f"PlantLayer: State saved into '{self.state_file}'")