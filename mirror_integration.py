# mirror_integration.py
# -*- coding: utf-8 -*-
"""
Mirror Integration Module for BioCortexAI
==========================================
Integrates the Digital Mirror concept into the LLM generation pipeline.

This module implements the "Predictive Mirror Loop":
1. Model generates response (hidden from user)
2. Response is mirrored (deictic swap JÁ ↔ TY) including context
3. Mirrored text is presented to model as simulated user input
4. Model generates "expected user response"
5. Expectation vectors are stored
6. Mirror context is cleared
7. Original response is shown to user
8. When user responds, actual vectors are compared with expected
9. Prediction error modulates PlantNet hormones

Supports two swap methods:
- "text": Regex-based text manipulation (fast, less accurate)
- "embedding": Vector space transformation (accurate, requires swap vector)

(c) 2025 OpenTechLab Jablonec nad Nisou s.r.o.
Author: Michal Seidl
"""

import torch
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import config

# Import from existing mirror_module (text-based swap and full phenomenological pipeline)
from mirror_module import (
    # Basic swap functions
    deikticky_swap,
    tokenize_cs,
    ZAMENA_JA,
    ZAMENA_TY,
    # Full phenomenological pipeline:
    # Φ (surface extractor)
    SurfaceOutput,
    analyzuj_povrch,
    priprav_rozsirene_rysy,
    # P_u (observer projection)
    ObserverProfil,
    vytvor_defaultni_profil,
    projektuj_vnimani,
    PercepcniVektor,
    # M_λ (mirror transformation)
    ZrcadloveOsy,
    aplikuj_styl,
    interpolate,  # Přidáno z prototypu
    # h (renderer)
    vytvor_lidsky_popis,
    sestav_agent_zpravu,
    # Main orchestration
    mirror_analyzuj,
    mirror_f,  # Zkrácená verze kompatibilní s prototypem
    MirrorConfig,  # Konfigurace z prototypu
    RozsirenaReflexe,
)

# Import embedding-space swap utilities
try:
    from swap_vector_utils import (
        SwapVectorManager,
        get_swap_manager,
        swap_hidden_states,
        load_swap_vector,
    )
    EMBEDDING_SWAP_AVAILABLE = True
except ImportError:
    EMBEDDING_SWAP_AVAILABLE = False
    print("Warning: swap_vector_utils not available, using text-based swap only")


# ==============================
# 1) Data structures
# ==============================

@dataclass
class ExpectationBuffer:
    """
    Buffer for storing expected response vectors.
    Holds the full sequence of hidden states from the expected response generation.
    """
    # Tensor of shape [seq_len, hidden_dim] - vectors for each token of expected response
    vectors: Optional[torch.Tensor] = None
    
    # Number of tokens in the expected response
    seq_len: int = 0
    
    # Mean vector for quick comparison
    mean_vector: Optional[np.ndarray] = None
    
    # Original response text (before swap)
    original_response: str = ""
    
    # Swapped context used for prediction
    swapped_context: str = ""
    
    # Expected response text (what model thinks user will say)
    expected_text: str = ""
    
    # Timestamp of creation
    timestamp: float = 0.0
    
    def clear(self):
        """Clear the buffer."""
        self.vectors = None
        self.seq_len = 0
        self.mean_vector = None
        self.original_response = ""
        self.swapped_context = ""
        self.expected_text = ""
        self.timestamp = 0.0
    
    def is_valid(self) -> bool:
        """Check if buffer contains valid expectation."""
        return self.vectors is not None and self.seq_len > 0


@dataclass
class PredictionResult:
    """Result of comparing expected vs actual user response."""
    # Prediction error (0 = perfect match, higher = more mismatch)
    prediction_error: float = 0.0
    
    # Cosine similarity (-1 to 1)
    cosine_similarity: float = 0.0
    
    # L2 distance (optional)
    l2_distance: Optional[float] = None
    
    # Hormone modulation deltas based on prediction error
    hormone_deltas: Dict[str, float] = field(default_factory=dict)
    
    # Quality category: "good", "neutral", "bad"
    quality: str = "neutral"
    
    # Debug info
    expected_seq_len: int = 0
    actual_seq_len: int = 0


# ==============================
# 2) Extended Context Swap
# ==============================

# Extended deictic patterns for context swap (including roles)
CONTEXT_ROLE_PATTERNS = [
    # user: -> model: swap
    (r'\buser:', '__ROLE_MODEL__'),
    (r'\bmodel:', '__ROLE_USER__'),
]

CONTEXT_ROLE_BACK = [
    (r'__ROLE_MODEL__', 'model:'),
    (r'__ROLE_USER__', 'user:'),
]


def swap_context_deixis(text: str, lam_deixis: float = 1.0) -> str:
    """
    Perform full deictic swap on context, including:
    - Personal pronouns (já/ty, mě/tě, etc.)
    - Role markers (user:/model:)
    
    Args:
        text: The text to swap
        lam_deixis: Intensity of swap (0.0 = none, 1.0 = full)
    
    Returns:
        Swapped text
    """
    if lam_deixis < 0.01:
        return text
    
    # First, swap role markers
    result = text
    for pattern, replacement in CONTEXT_ROLE_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    for pattern, replacement in CONTEXT_ROLE_BACK:
        result = re.sub(pattern, replacement, result)
    
    # Then, swap pronouns using mirror_module function
    result = deikticky_swap(result, lam_deixis)
    
    return result


def prepare_mirror_context(
    conversation_tokens: List[int],
    response_text: str,
    tokenizer,
    max_context_tokens: int = None
) -> Tuple[str, List[int]]:
    """
    Prepare the mirrored context for expectation generation.
    
    1. Takes recent conversation context (limited by max_context_tokens)
    2. Swaps all deictic markers
    3. Appends the swapped model response as if it were user input
    
    Args:
        conversation_tokens: Full conversation token history
        response_text: The model's response (not yet shown to user)
        tokenizer: SentencePiece tokenizer
        max_context_tokens: Maximum tokens to include from context
    
    Returns:
        Tuple of (swapped_context_text, swapped_context_tokens)
    """
    if max_context_tokens is None:
        max_context_tokens = config.MIRROR_CONTEXT_LAST_TOKENS
    
    # Get recent context tokens
    context_tokens = conversation_tokens[-max_context_tokens:] if len(conversation_tokens) > max_context_tokens else conversation_tokens
    
    # Decode context to text
    context_text = tokenizer.decode(context_tokens)
    
    # Swap the context
    swapped_context = swap_context_deixis(context_text, config.MIRROR_LAMBDA_DEIXIS)
    
    # Swap the response and format as user input
    swapped_response = swap_context_deixis(response_text, config.MIRROR_LAMBDA_DEIXIS)
    
    # Create the full mirrored prompt
    # The swapped response becomes the "user" input, model should respond
    full_mirrored = f"{swapped_context} user: {swapped_response} model:"
    
    # Tokenize
    mirrored_tokens = [tokenizer.bos_id()] + tokenizer.encode_as_ids(full_mirrored)
    
    return full_mirrored, mirrored_tokens


# ==============================
# 3) Vector Comparison
# ==============================

def compute_prediction_error(
    expected_vectors: torch.Tensor,
    actual_vectors: torch.Tensor,
    method: str = None
) -> PredictionResult:
    """
    Compare expected vs actual response vectors.
    
    Args:
        expected_vectors: Tensor [seq_len_exp, hidden_dim] or [hidden_dim]
        actual_vectors: Tensor [seq_len_act, hidden_dim] or [hidden_dim]
        method: Comparison method ("cosine", "l2", "combined")
    
    Returns:
        PredictionResult with error metrics and hormone deltas
    """
    if method is None:
        method = config.MIRROR_COMPARISON_METHOD
    
    result = PredictionResult()
    
    # Handle different tensor shapes - reduce to mean vectors if needed
    if expected_vectors.dim() == 2:
        exp_mean = expected_vectors.mean(dim=0)
        result.expected_seq_len = expected_vectors.shape[0]
    else:
        exp_mean = expected_vectors
        result.expected_seq_len = 1
    
    if actual_vectors.dim() == 2:
        act_mean = actual_vectors.mean(dim=0)
        result.actual_seq_len = actual_vectors.shape[0]
    else:
        act_mean = actual_vectors
        result.actual_seq_len = 1
    
    # Ensure same device
    if exp_mean.device != act_mean.device:
        act_mean = act_mean.to(exp_mean.device)
    
    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        exp_mean.unsqueeze(0), 
        act_mean.unsqueeze(0),
        dim=1
    ).item()
    
    result.cosine_similarity = cos_sim
    
    # Cosine distance (0 = identical, 2 = opposite)
    cosine_distance = 1.0 - cos_sim
    
    if method == "cosine":
        result.prediction_error = cosine_distance
    
    elif method == "l2":
        l2_dist = torch.nn.functional.pairwise_distance(
            exp_mean.unsqueeze(0),
            act_mean.unsqueeze(0)
        ).item()
        result.l2_distance = l2_dist
        # Normalize L2 to roughly same scale as cosine distance
        result.prediction_error = min(l2_dist / 10.0, 2.0)
    
    elif method == "combined":
        l2_dist = torch.nn.functional.pairwise_distance(
            exp_mean.unsqueeze(0),
            act_mean.unsqueeze(0)
        ).item()
        result.l2_distance = l2_dist
        # Weighted combination
        result.prediction_error = 0.7 * cosine_distance + 0.3 * min(l2_dist / 10.0, 2.0)
    
    else:
        result.prediction_error = cosine_distance
    
    # Determine quality category and hormone deltas
    if result.prediction_error < config.MIRROR_ERROR_THRESHOLD_LOW:
        result.quality = "good"
        result.hormone_deltas = dict(config.MIRROR_GOOD_PREDICTION)
    elif result.prediction_error > config.MIRROR_ERROR_THRESHOLD_HIGH:
        result.quality = "bad"
        result.hormone_deltas = dict(config.MIRROR_BAD_PREDICTION)
    else:
        result.quality = "neutral"
        result.hormone_deltas = dict(config.MIRROR_NEUTRAL_PREDICTION)
    
    return result


# ==============================
# 4) MirrorIntegration Class
# ==============================

class MirrorIntegration:
    """
    Main class for integrating Digital Mirror into BioCortexAI.
    
    Manages the prediction loop and interfaces with PlantNet.
    Supports both text-based and embedding-space swap methods.
    """
    
    def __init__(self):
        self.expectation_buffer = ExpectationBuffer()
        self.enabled = config.USE_MIRROR_MODULE
        self.last_prediction_result: Optional[PredictionResult] = None
        
        # Statistics
        self.total_predictions = 0
        self.good_predictions = 0
        self.bad_predictions = 0
        
        # Swap method
        self.swap_method = getattr(config, 'MIRROR_SWAP_METHOD', 'text')
        self.swap_manager: Optional[SwapVectorManager] = None
        self._swap_initialized = False
        
        if config.MIRROR_VERBOSE:
            print(f"MirrorIntegration: Initialized (swap_method={self.swap_method})")
    
    def initialize_embedding_swap(self, model, tokenizer, device: str):
        """
        Initialize embedding-space swap with the model.
        Must be called before using embedding swap method.
        
        Args:
            model: The Transformer model
            tokenizer: SentencePiece tokenizer
            device: torch device string
        """
        if not EMBEDDING_SWAP_AVAILABLE:
            print("Warning: Embedding swap not available, falling back to text swap")
            self.swap_method = "text"
            return
        
        self.swap_manager = SwapVectorManager(model, tokenizer, device)
        try:
            self.swap_manager.initialize(force_derive=False)
            self._swap_initialized = True
            if config.MIRROR_VERBOSE or getattr(config, 'MIRROR_DEBUG', False):
                print("MirrorIntegration: Embedding swap vector initialized")
        except Exception as e:
            print(f"Warning: Failed to initialize embedding swap: {e}")
            print("Falling back to text-based swap")
            self.swap_method = "text"
    
    def swap_hidden_embedding(
        self, 
        hidden_states: torch.Tensor,
        intensity: float = None
    ) -> torch.Tensor:
        """
        Apply embedding-space swap to hidden states.
        
        Args:
            hidden_states: Tensor [batch, seq_len, hidden_dim]
            intensity: Swap intensity (default: MIRROR_LAMBDA_DEIXIS)
        
        Returns:
            Swapped hidden states
        """
        if not self._swap_initialized or self.swap_manager is None:
            return hidden_states
        
        if intensity is None:
            intensity = config.MIRROR_LAMBDA_DEIXIS
        
        return self.swap_manager.swap(hidden_states, intensity, "self_to_other")
    
    def get_perspective_score(self, hidden_states: torch.Tensor) -> float:
        """
        Get perspective score for hidden states.
        Positive = SELF perspective, Negative = OTHER perspective.
        """
        if self.swap_manager is not None:
            return self.swap_manager.get_perspective_score(hidden_states)
        return 0.0
    
    def analyze_phenomenologically(
        self,
        text: str,
        observer_name: str = "LLM",
        mode: str = "agent",
        lam_deixis: float = None,
        lam_styl: float = None,
    ) -> Optional[RozsirenaReflexe]:
        """
        Perform full phenomenological analysis of text output.
        
        Implements the complete mirror function f(O_t; u, C, λ):
        1. Φ (surface extractor) - extract surface features
        2. P_u (observer projection) - project to perceptual space
        3. M_λ (mirror transformation) - apply deictic and style transforms
        4. h (renderer) - render the reflection
        
        Args:
            text: The model's output text to analyze
            observer_name: Who is observing ("LLM", "human", "critic", etc.)
            mode: Output mode ("agent" for structured, "human" for readable)
            lam_deixis: Deictic swap intensity (default from config)
            lam_styl: Style transformation intensity (default from config)
        
        Returns:
            RozsirenaReflexe containing the full phenomenological reflection
        """
        if not text:
            return None
        
        if lam_deixis is None:
            lam_deixis = config.MIRROR_LAMBDA_DEIXIS
        if lam_styl is None:
            lam_styl = config.MIRROR_LAMBDA_STYL
        
        # Create surface output structure
        surface = SurfaceOutput(text=text, next_token_probs=None)
        
        # Create mirror axes
        axes = ZrcadloveOsy(lam_deixis=lam_deixis, lam_styl=lam_styl)
        
        # Create observer profile based on observer_name
        if observer_name == "LLM":
            # Model looking at itself - neutral projection
            profil = vytvor_defaultni_profil()
        elif observer_name == "critic":
            # Critical observer - emphasize formality and assertiveness issues
            profil = ObserverProfil(
                jmeno="kritik",
                w_formalita={"avg_sentence_len": 0.5, "long_word_ratio": 0.6, "politeness_hits": 0.3},
                w_vrelost={"politeness_hits": 0.4, "hedging_hits": -0.2},
                w_asertivita={"hedging_hits": -0.8, "emphasis_hits": 0.6},
            )
        elif observer_name == "human":
            # Human observer - emphasize clarity and warmth
            profil = ObserverProfil(
                jmeno="clovek",
                w_formalita={"avg_sentence_len": 0.3, "long_word_ratio": 0.4},
                w_vrelost={"politeness_hits": 0.7, "hedging_hits": 0.3, "emphasis_hits": -0.2},
                w_asertivita={"hedging_hits": -0.5, "emphasis_hits": 0.4},
            )
        else:
            profil = vytvor_defaultni_profil()
        
        # Run full phenomenological analysis
        try:
            reflection = mirror_analyzuj(
                povrch=surface,
                osy=axes,
                jmeno_pozorovatele=observer_name,
                rezim=mode,
                profil=profil,
            )
            return reflection
        except Exception as e:
            if config.MIRROR_VERBOSE or getattr(config, 'MIRROR_DEBUG', False):
                print(f"Error in phenomenological analysis: {e}")
            return None
    
    def get_reflection_summary(self, reflection: RozsirenaReflexe) -> Dict[str, Any]:
        """
        Extract key metrics from phenomenological reflection.
        
        Returns dict with:
        - formality, warmth, assertiveness (perceptual dimensions)
        - self_focus, meta_ratio (perspective metrics)  
        - paraphrase (swapped text)
        - agent_report (structured report for downstream use)
        """
        if reflection is None:
            return {}
        
        return {
            "formality": reflection.vektor.formalita if reflection.vektor else 0.0,
            "warmth": reflection.vektor.vrelost if reflection.vektor else 0.0,
            "assertiveness": reflection.vektor.asertivita if reflection.vektor else 0.0,
            "deictic_role": reflection.vektor.deikticka_role if reflection.vektor else "unknown",
            "paraphrase": reflection.parafraze,
            "description": reflection.popis,
            "agent_report": reflection.agent_zprava,
            "lambda_deixis": reflection.lambda_pouzite.get("deixis", 0.0) if reflection.lambda_pouzite else 0.0,
            "lambda_style": reflection.lambda_pouzite.get("styl", 0.0) if reflection.lambda_pouzite else 0.0,
            "observer": reflection.pozorovatel,
        }
    
    def store_expectation(
        self,
        response_text: str,
        expected_text: str,
        expected_hidden_states: torch.Tensor,
        swapped_context: str = ""
    ):
        """
        Store the expected response vectors in the buffer.
        
        Args:
            response_text: Original model response
            expected_text: Generated expected user response
            expected_hidden_states: Hidden states from expectation generation
                                   Shape: [batch, seq_len, hidden_dim]
            swapped_context: The swapped context used
        """
        import time
        
        # Extract vectors - we want the full response sequence, not just last token
        # expected_hidden_states shape: [1, seq_len, hidden_dim]
        if expected_hidden_states.dim() == 3:
            vectors = expected_hidden_states[0].detach().cpu()  # [seq_len, hidden_dim]
        else:
            vectors = expected_hidden_states.detach().cpu()
        
        self.expectation_buffer.vectors = vectors
        self.expectation_buffer.seq_len = vectors.shape[0]
        self.expectation_buffer.mean_vector = vectors.mean(dim=0).numpy()
        self.expectation_buffer.original_response = response_text
        self.expectation_buffer.expected_text = expected_text
        self.expectation_buffer.swapped_context = swapped_context
        self.expectation_buffer.timestamp = time.time()
        
        if config.MIRROR_VERBOSE:
            print(f"MirrorIntegration: Stored expectation with {self.expectation_buffer.seq_len} tokens")
            print(f"  Expected text: {expected_text[:100]}...")
    
    def compare_with_actual(
        self,
        actual_hidden_states: torch.Tensor,
        actual_text: str = ""
    ) -> Optional[PredictionResult]:
        """
        Compare stored expectation with actual user response.
        
        Args:
            actual_hidden_states: Hidden states from actual user input processing
                                  Shape: [batch, seq_len, hidden_dim]
            actual_text: The actual user response text (for logging)
        
        Returns:
            PredictionResult or None if no expectation stored
        """
        if not self.expectation_buffer.is_valid():
            if config.MIRROR_VERBOSE:
                print("MirrorIntegration: No valid expectation to compare")
            return None
        
        # Extract actual vectors
        if actual_hidden_states.dim() == 3:
            actual_vectors = actual_hidden_states[0].detach()  # [seq_len, hidden_dim]
        else:
            actual_vectors = actual_hidden_states.detach()
        
        # Compute prediction error
        result = compute_prediction_error(
            self.expectation_buffer.vectors.to(actual_vectors.device),
            actual_vectors,
            config.MIRROR_COMPARISON_METHOD
        )
        
        self.last_prediction_result = result
        self.total_predictions += 1
        
        if result.quality == "good":
            self.good_predictions += 1
        elif result.quality == "bad":
            self.bad_predictions += 1
        
        if config.MIRROR_VERBOSE:
            print(f"MirrorIntegration: Prediction comparison")
            print(f"  Error: {result.prediction_error:.4f}")
            print(f"  Cosine similarity: {result.cosine_similarity:.4f}")
            print(f"  Quality: {result.quality}")
            print(f"  Expected: {self.expectation_buffer.expected_text[:50]}...")
            print(f"  Actual: {actual_text[:50]}...")
        
        return result
    
    def apply_to_plant_net(
        self,
        plant_layer,
        prediction_result: PredictionResult
    ):
        """
        Apply prediction error to PlantNet hormones.
        
        Args:
            plant_layer: The PlantLayer instance
            prediction_result: Result from compare_with_actual
        """
        if plant_layer is None or prediction_result is None:
            return
        
        # Get the global hormone state
        network = plant_layer.network
        
        # Apply hormone deltas to all cells with some influence
        for row in network.cells:
            for cell in row:
                for hormone, delta in prediction_result.hormone_deltas.items():
                    if hormone in cell.hormones:
                        cell.hormones[hormone] += delta
                        # Clamp to valid range
                        cell.hormones[hormone] = max(
                            cell.HORMONE_RANGE[0],
                            min(cell.HORMONE_RANGE[1], cell.hormones[hormone])
                        )
        
        # Optionally store in associative memory
        if config.MIRROR_STORE_EXPECTATIONS and self.expectation_buffer.is_valid():
            # Store the prediction context as a memory
            for row in network.cells:
                for cell in row:
                    if hasattr(cell, 'memory') and self.expectation_buffer.mean_vector is not None:
                        reaction = {
                            "prediction_error": prediction_result.prediction_error,
                            "quality": prediction_result.quality,
                            "cosine_sim": prediction_result.cosine_similarity,
                        }
                        cell.memory.add(
                            embedding=self.expectation_buffer.mean_vector,
                            reaction=reaction,
                            hormones={k: float(v) for k, v in cell.hormones.items()},
                            weight=config.MIRROR_EXPECTATION_MEMORY_WEIGHT
                        )
                        break  # Only need to store once
                break
        
        if config.MIRROR_VERBOSE:
            print(f"MirrorIntegration: Applied hormone deltas: {prediction_result.hormone_deltas}")
    
    def clear_expectation(self):
        """Clear the expectation buffer after use."""
        self.expectation_buffer.clear()
        if config.MIRROR_VERBOSE:
            print("MirrorIntegration: Cleared expectation buffer")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        accuracy = 0.0
        if self.total_predictions > 0:
            accuracy = self.good_predictions / self.total_predictions
        
        return {
            "total_predictions": self.total_predictions,
            "good_predictions": self.good_predictions,
            "bad_predictions": self.bad_predictions,
            "neutral_predictions": self.total_predictions - self.good_predictions - self.bad_predictions,
            "prediction_accuracy": accuracy,
            "last_error": self.last_prediction_result.prediction_error if self.last_prediction_result else None,
        }


# ==============================
# 5) Singleton instance
# ==============================

# Global instance for easy access
mirror_integration = MirrorIntegration()


# ==============================
# 6) Demo / Test
# ==============================

if __name__ == "__main__":
    print("=== Mirror Integration Test ===\n")
    
    # Test context swap
    test_context = """user: Jaký je smysl života?
model: Smysl života je subjektivní a závisí na každém člověku.
user: A co ty si myslíš?
model: Já si myslím, že je důležité hledat štěstí a naplnění."""
    
    print("Original context:")
    print(test_context)
    print("\nSwapped context:")
    print(swap_context_deixis(test_context, 1.0))
    
    # Test prediction error computation
    print("\n=== Prediction Error Test ===")
    expected = torch.randn(10, 256)  # 10 tokens, 256 dim
    
    # Similar vectors (good prediction)
    actual_good = expected + torch.randn_like(expected) * 0.1
    result_good = compute_prediction_error(expected, actual_good)
    print(f"Good prediction: error={result_good.prediction_error:.4f}, quality={result_good.quality}")
    
    # Different vectors (bad prediction)
    actual_bad = torch.randn(15, 256)
    result_bad = compute_prediction_error(expected, actual_bad)
    print(f"Bad prediction: error={result_bad.prediction_error:.4f}, quality={result_bad.quality}")
