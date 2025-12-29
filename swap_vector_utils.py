# swap_vector_utils.py
# -*- coding: utf-8 -*-
"""
Swap Vector Utilities for BioCortexAI
=====================================
Derives and manages the deictic swap vector for perspective transformation
in embedding space.

The swap vector represents the semantic direction from "self" (JÁ/I) to 
"other" (TY/YOU) in the model's hidden state space. This allows for 
grammatically and semantically correct perspective swapping without
regex-based text manipulation.

(c) 2025 OpenTechLab Jablonec nad Nisou s.r.o.
Author: Michal Seidl
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Optional
import config


# ==============================
# 1) Contrastive Pairs for Czech
# ==============================

# Pairs of sentences: (SELF perspective, OTHER perspective)
# These should have identical meaning, differing only in perspective

CZECH_DEIXIS_PAIRS = [
    # Basic pronouns
    ("Já jsem tady.", "Ty jsi tady."),
    ("Mám se dobře.", "Máš se dobře."),
    ("Jsem spokojený.", "Jsi spokojený."),
    ("Je mi smutno.", "Je ti smutno."),
    ("Mám radost.", "Máš radost."),
    
    # Possessives
    ("Moje odpověď je jasná.", "Tvoje odpověď je jasná."),
    ("Můj názor je jiný.", "Tvůj názor je jiný."),
    ("To je moje myšlenka.", "To je tvoje myšlenka."),
    
    # Accusative
    ("Myslím na tebe.", "Myslíš na mě."),
    ("Vidím tě.", "Vidíš mě."),
    ("Slyším tě dobře.", "Slyšíš mě dobře."),
    
    # Dative
    ("Pomohl jsem ti.", "Pomohl jsi mi."),
    ("Dal jsem ti odpověď.", "Dal jsi mi odpověď."),
    ("Řekl jsem ti pravdu.", "Řekl jsi mi pravdu."),
    
    # Instrumental
    ("Jdu s tebou.", "Jdeš se mnou."),
    ("Mluvím s tebou.", "Mluvíš se mnou."),
    
    # Complex sentences
    ("Já si myslím, že to je správné.", "Ty si myslíš, že to je správné."),
    ("Snažím se ti porozumět.", "Snažíš se mi porozumět."),
    ("Chtěl bych ti pomoci.", "Chtěl bys mi pomoci."),
    ("Nevím, co ti mám říct.", "Nevíš, co mi máš říct."),
    
    # Conversational context (model/user)
    ("Já jako model odpovídám na tvou otázku.", "Ty jako uživatel odpovídáš na mou otázku."),
    ("Snažím se ti vysvětlit koncept.", "Snažíš se mi vysvětlit koncept."),
    ("Ptám se tě na tvůj názor.", "Ptáš se mě na můj názor."),
    
    # Questions
    ("Co si o tom myslím?", "Co si o tom myslíš?"),
    ("Jak se cítím?", "Jak se cítíš?"),
    ("Co mám dělat?", "Co máš dělat?"),
    
    # Negations
    ("Nerozumím ti.", "Nerozumíš mi."),
    ("Nevěřím ti.", "Nevěříš mi."),
    ("Neslyším tě.", "Neslyšíš mě."),
    
    # Past tense
    ("Viděl jsem tě včera.", "Viděl jsi mě včera."),
    ("Slyšel jsem, co jsi řekl.", "Slyšel jsi, co jsem řekl."),
    ("Pamatuju si, co jsi mi řekl.", "Pamatuješ si, co jsem ti řekl."),
    
    # Future tense
    ("Pomůžu ti zítra.", "Pomůžeš mi zítra."),
    ("Zavolám ti večer.", "Zavoláš mi večer."),
    ("Napíšu ti email.", "Napíšeš mi email."),
]

# Role swap pairs for conversation context
ROLE_SWAP_PAIRS = [
    ("user: Ahoj model: Ahoj!", "model: Ahoj user: Ahoj!"),
    ("user: Jak se máš? model: Dobře, díky.", "model: Jak se máš? user: Dobře, díky."),
]


# ==============================
# 2) Swap Vector Derivation
# ==============================

@torch.no_grad()
def derive_swap_vector(
    model,
    tokenizer,
    device: str,
    pairs: List[Tuple[str, str]] = None,
    pooling: str = "mean",
    normalize: bool = True,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Derive the swap vector from contrastive sentence pairs.
    
    The swap vector represents the direction from SELF to OTHER perspective
    in the model's hidden state space.
    
    Args:
        model: The Transformer model
        tokenizer: SentencePiece tokenizer
        device: torch device
        pairs: List of (self_sentence, other_sentence) pairs. 
               If None, uses CZECH_DEIXIS_PAIRS
        pooling: How to aggregate hidden states ("mean", "last", "first")
        normalize: Whether to normalize the final vector
        scale: Scale factor for the normalized vector
    
    Returns:
        swap_vector: Tensor of shape [hidden_dim]
    """
    if pairs is None:
        pairs = CZECH_DEIXIS_PAIRS
    
    model.eval()
    differences = []
    
    for self_sent, other_sent in pairs:
        try:
            # Tokenize
            self_tokens = [tokenizer.bos_id()] + tokenizer.encode_as_ids(self_sent)
            other_tokens = [tokenizer.bos_id()] + tokenizer.encode_as_ids(other_sent)
            
            # Get hidden states
            self_input = torch.tensor([self_tokens], dtype=torch.long, device=device)
            other_input = torch.tensor([other_tokens], dtype=torch.long, device=device)
            
            _, self_hidden = model(self_input, return_hidden=True)
            _, other_hidden = model(other_input, return_hidden=True)
            
            # Pool hidden states to single vector
            if pooling == "mean":
                self_vec = self_hidden.mean(dim=1).squeeze()
                other_vec = other_hidden.mean(dim=1).squeeze()
            elif pooling == "last":
                self_vec = self_hidden[:, -1, :].squeeze()
                other_vec = other_hidden[:, -1, :].squeeze()
            elif pooling == "first":
                self_vec = self_hidden[:, 0, :].squeeze()
                other_vec = other_hidden[:, 0, :].squeeze()
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
            
            # Compute difference: SELF - OTHER
            diff = self_vec - other_vec
            differences.append(diff)
            
        except Exception as e:
            if getattr(config, 'MIRROR_VERBOSE', False):
                print(f"Warning: Skipping pair due to error: {e}")
            continue
    
    if not differences:
        raise ValueError("No valid pairs for deriving swap vector")
    
    # Average direction across all pairs
    swap_vector = torch.stack(differences).mean(dim=0)
    
    # Normalize and scale
    if normalize:
        norm = swap_vector.norm()
        if norm > 0:
            swap_vector = swap_vector / norm * scale
    
    return swap_vector


def save_swap_vector(swap_vector: torch.Tensor, path: str = None):
    """Save swap vector to file."""
    if path is None:
        path = getattr(config, 'SWAP_VECTOR_PATH', 'swap_vector.pt')
    
    torch.save({
        'swap_vector': swap_vector.cpu(),
        'hidden_dim': swap_vector.shape[0],
    }, path)
    
    print(f"Swap vector saved to: {path}")
    print(f"  Shape: {swap_vector.shape}")
    print(f"  Norm: {swap_vector.norm().item():.4f}")


def load_swap_vector(path: str = None, device: str = 'cpu') -> Optional[torch.Tensor]:
    """Load swap vector from file."""
    if path is None:
        path = getattr(config, 'SWAP_VECTOR_PATH', 'swap_vector.pt')
    
    if not os.path.exists(path):
        return None
    
    data = torch.load(path, map_location=device)
    swap_vector = data['swap_vector']
    
    if getattr(config, 'MIRROR_VERBOSE', False):
        print(f"Loaded swap vector from: {path}")
        print(f"  Shape: {swap_vector.shape}")
    
    return swap_vector


# ==============================
# 3) Hidden State Transformation
# ==============================

def swap_hidden_states(
    hidden_states: torch.Tensor,
    swap_vector: torch.Tensor,
    intensity: float = 1.0,
    direction: str = "self_to_other"
) -> torch.Tensor:
    """
    Transform hidden states by applying the swap vector.
    
    Args:
        hidden_states: Tensor of shape [batch, seq_len, hidden_dim]
        swap_vector: Tensor of shape [hidden_dim]
        intensity: Strength of transformation (0.0 = none, 1.0 = full)
        direction: "self_to_other" or "other_to_self"
    
    Returns:
        Transformed hidden states with same shape
    """
    if intensity <= 0:
        return hidden_states
    
    # Ensure swap_vector is on same device
    swap_vec = swap_vector.to(hidden_states.device)
    
    # Reshape for broadcasting: [1, 1, hidden_dim]
    swap_vec = swap_vec.unsqueeze(0).unsqueeze(0)
    
    # Apply transformation
    if direction == "self_to_other":
        # Moving from SELF perspective toward OTHER perspective
        # SELF - swap_vector = OTHER
        return hidden_states - intensity * swap_vec
    else:
        # Moving from OTHER perspective toward SELF perspective
        # OTHER + swap_vector = SELF
        return hidden_states + intensity * swap_vec


def compute_perspective_score(
    hidden_states: torch.Tensor,
    swap_vector: torch.Tensor
) -> float:
    """
    Compute how much the hidden states align with SELF vs OTHER perspective.
    
    Returns:
        Score in range [-1, 1]:
        - Positive = more SELF-like
        - Negative = more OTHER-like
        - Zero = neutral
    """
    swap_vec = swap_vector.to(hidden_states.device)
    
    # Mean hidden state
    mean_hidden = hidden_states.mean(dim=(0, 1))
    
    # Cosine similarity with swap direction
    cos_sim = torch.nn.functional.cosine_similarity(
        mean_hidden.unsqueeze(0),
        swap_vec.unsqueeze(0),
        dim=1
    )
    
    return cos_sim.item()


# ==============================
# 4) Swap Vector Manager
# ==============================

class SwapVectorManager:
    """
    Manages the swap vector lifecycle: derivation, storage, and application.
    """
    
    def __init__(self, model=None, tokenizer=None, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.swap_vector: Optional[torch.Tensor] = None
        self._initialized = False
    
    def initialize(self, force_derive: bool = False):
        """
        Initialize the swap vector - load from file or derive.
        
        Args:
            force_derive: If True, derive new vector even if file exists
        """
        vector_path = getattr(config, 'SWAP_VECTOR_PATH', 'swap_vector.pt')
        
        if not force_derive:
            # Try to load existing vector
            self.swap_vector = load_swap_vector(vector_path, self.device)
            
            if self.swap_vector is not None:
                self._initialized = True
                return
        
        # Derive new vector
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer required for deriving swap vector")
        
        print("Deriving swap vector from contrastive pairs...")
        self.swap_vector = derive_swap_vector(
            self.model,
            self.tokenizer,
            self.device,
            normalize=True,
            scale=getattr(config, 'SWAP_VECTOR_SCALE', 0.5)
        )
        
        # Save for future use
        save_swap_vector(self.swap_vector, vector_path)
        self._initialized = True
    
    def swap(
        self,
        hidden_states: torch.Tensor,
        intensity: float = None,
        direction: str = "self_to_other"
    ) -> torch.Tensor:
        """
        Apply perspective swap to hidden states.
        """
        if not self._initialized or self.swap_vector is None:
            raise RuntimeError("SwapVectorManager not initialized. Call initialize() first.")
        
        if intensity is None:
            intensity = getattr(config, 'MIRROR_LAMBDA_DEIXIS', 1.0)
        
        return swap_hidden_states(
            hidden_states,
            self.swap_vector,
            intensity,
            direction
        )
    
    def get_perspective_score(self, hidden_states: torch.Tensor) -> float:
        """Get perspective score for hidden states."""
        if self.swap_vector is None:
            return 0.0
        return compute_perspective_score(hidden_states, self.swap_vector)


# Global manager instance
swap_manager: Optional[SwapVectorManager] = None


def get_swap_manager() -> SwapVectorManager:
    """Get or create global swap manager."""
    global swap_manager
    if swap_manager is None:
        swap_manager = SwapVectorManager()
    return swap_manager


# ==============================
# 5) CLI for deriving swap vector
# ==============================

if __name__ == "__main__":
    import argparse
    import sentencepiece as spm
    from model import Transformer
    
    parser = argparse.ArgumentParser(description="Derive swap vector for BioCortexAI")
    parser.add_argument('--model', type=str, default='finetuned_model.pth',
                        help='Path to model file')
    parser.add_argument('--output', type=str, default='swap_vector.pt',
                        help='Output path for swap vector')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scale factor for normalized vector')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    model_data = torch.load(args.model, map_location=device)
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
    model.eval()
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(f"{config.model_prefix}.model")
    
    # Derive swap vector
    print(f"\nDeriving swap vector from {len(CZECH_DEIXIS_PAIRS)} contrastive pairs...")
    swap_vector = derive_swap_vector(
        model, sp, device,
        normalize=True,
        scale=args.scale
    )
    
    # Save
    save_swap_vector(swap_vector, args.output)
    
    # Test
    print("\n=== Testing swap vector ===")
    test_pairs = [
        ("Já jsem model.", "Ty jsi model."),
        ("Pomůžu ti.", "Pomůžeš mi."),
    ]
    
    for self_sent, other_sent in test_pairs:
        self_tokens = [sp.bos_id()] + sp.encode_as_ids(self_sent)
        other_tokens = [sp.bos_id()] + sp.encode_as_ids(other_sent)
        
        self_input = torch.tensor([self_tokens], dtype=torch.long, device=device)
        other_input = torch.tensor([other_tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            _, self_hidden = model(self_input, return_hidden=True)
            _, other_hidden = model(other_input, return_hidden=True)
        
        # Compute perspective scores
        self_score = compute_perspective_score(self_hidden, swap_vector)
        other_score = compute_perspective_score(other_hidden, swap_vector)
        
        print(f"\n'{self_sent}'")
        print(f"  Perspective score: {self_score:+.4f} (should be positive)")
        print(f"'{other_sent}'")
        print(f"  Perspective score: {other_score:+.4f} (should be negative)")
    
    print("\n✅ Swap vector derivation complete!")
