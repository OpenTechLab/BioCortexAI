# config.py
from enum import Enum

# -----------------
# Helps & Tips
# -----------------
# This GPT configuration (tini 13M model) requires:
#     a pre-training dataset of at least 500MB of high-quality literary and highly versatile text.
#     dataset for fine-tuning at least 20,000 Q/A blocks user/model. See sample dataset for correct formatting.
# We strongly recommend CUDA for training and operation. 
#     This tini 13M model can be trained on less than 4GB of VRAM.
# Such a tiny GPT model is suitable for experimental and study purposes. 
# A model usable for real deployment must have significantly more parameters;
#     (minimum configuration mentioned).
#
# -----------------
# Parameters of the tokenizer
# -----------------
vocab_size = 32000                             # Set vocab size, very small model - 64000 may be better (follow your dataset)
model_prefix = 'spm_model'
model_type = 'bpe'

raw_data_dir = "data/raw_data"                 # Raw data directory
preprocessed_text_path = "training_corpus.txt" # preprocess_corpus.py output
chunked_binary_path = "training_corpus.bin"    # chunk_corpus.py output

# -----------------
# Model architecture parameters (shared across all phases).
# -----------------
# These parameters define how large the model is and what its internal structure looks like.
# They must be the same during pre-training, fine-tuning, and generation.

embedding_dim = 256           # very small model - 2048 is better
n_layers = 8                  # very small model - 32 is better
n_heads = 8                   # very small model - 32 is better
n_kv_heads = 4                # very small model - 16 is better
ff_dim_multiplier = 2.666
dropout = 0.1


# -----------------
# Parameters for pre-training (pretrain.py)
# -----------------
pretrain_checkpoint_dir = 'checkpoints/base_model'
pretrain_num_epochs = 110       # Adjust according to the situation
pretrain_learning_rate = 3e-4  # Max LR
pretrain_min_lr = 1e-5         # Min LR on the end
pretrain_warmup_iters = 900    # warmup steps, very small model - 2000 is better
pretrain_batch_size = 6        # by your HW
pretrain_max_seq_len = 512     # very small model - 4096 is better

# -----------------
# Parameters for fine-tuning (finetune.py)
# -----------------
finetune_data_path = 'data/CZ_QA_MIKRO.txt'   #replace with yours, see example
base_model_path = 'checkpoints/base_model/best_checkpoint.pt'
finetune_checkpoint_dir = 'checkpoints/finetuned_model'
finetune_num_epochs = 100       # Adjust according to the situation
finetune_learning_rate = 1e-5  # Max LR
finetune_min_lr = 1e-6         # Min LR
finetune_warmup_iters = 50     # warmup steps, very small model - 900 is better

finetune_batch_size = 2        # by your HW

# -----------------
# Parameters for generator (generate.py)
# -----------------
generator_checkpoint_path = 'checkpoints/finetuned_model/latest_checkpoint.pt'

num_workers = 0
grad_clip = 1.0
resume_from_checkpoint = True

# -----------------
# Parameters for plant network (plant_network.py)
# -----------------

# --- MAIN SWITCH for ACTIVATING/DEACTIVATING the entire functionality. ---
USE_PLANT_NETWORK = True # always False during pre-training and fine-tuning.
# -------------------------------------------------------------------------

# File for saving and loading the state of the plant network.
PLANT_NET_STATE_FILE = "plant_net_state.json"

# Definition of personalities (Enum)
class Personality(Enum):
    OPTIMISTIC = 1 # optimistický
    CAUTIOUS = 2   # opatrný
    CURIOUS = 3    # zvědavý

# The tissue map defines the structure and behavior of the plant network.
# Each element is a tuple (CellType, Personality).

PLANT_TISSUE_MAP = [
    # r0 (top)
    [('SensorCell',     Personality.CAUTIOUS),  ('SensorCell',     Personality.CURIOUS),   ('StructuralCell', Personality.OPTIMISTIC), ('StructuralCell', Personality.CAUTIOUS),
     ('MemoryCell',     Personality.CAUTIOUS),  ('MemoryCell',     Personality.OPTIMISTIC),('MemoryCell',     Personality.CAUTIOUS),
     ('StructuralCell', Personality.OPTIMISTIC),('StructuralCell', Personality.OPTIMISTIC),('SensorCell',     Personality.CURIOUS),   ('SensorCell',     Personality.CAUTIOUS)],

    # r1
    [('SensorCell',     Personality.CURIOUS),   ('SensorCell',     Personality.CURIOUS),   ('StructuralCell', Personality.OPTIMISTIC), ('StructuralCell', Personality.OPTIMISTIC),
     ('MemoryCell',     Personality.CAUTIOUS),  ('MemoryCell',     Personality.CAUTIOUS),  ('MemoryCell',     Personality.OPTIMISTIC),
     ('StructuralCell', Personality.OPTIMISTIC),('StructuralCell', Personality.CAUTIOUS),  ('SensorCell',     Personality.CURIOUS),    ('SensorCell',     Personality.OPTIMISTIC)],

    # r2
    [('SensorCell',     Personality.OPTIMISTIC),('SensorCell',     Personality.CURIOUS),   ('StructuralCell', Personality.OPTIMISTIC), ('StructuralCell', Personality.OPTIMISTIC),
     ('MemoryCell',     Personality.CAUTIOUS),  ('PlantCell',      Personality.OPTIMISTIC),('MemoryCell',     Personality.CAUTIOUS),
     ('StructuralCell', Personality.OPTIMISTIC),('StructuralCell', Personality.CAUTIOUS),  ('SensorCell',     Personality.CURIOUS),    ('SensorCell',     Personality.CURIOUS)],

    # r3
    [('SensorCell',     Personality.CURIOUS),   ('SensorCell',     Personality.CAUTIOUS),  ('StructuralCell', Personality.CAUTIOUS),   ('StructuralCell', Personality.OPTIMISTIC),
     ('MemoryCell',     Personality.CAUTIOUS),  ('PlantCell',      Personality.CAUTIOUS),  ('MemoryCell',     Personality.OPTIMISTIC),
     ('StructuralCell', Personality.OPTIMISTIC),('StructuralCell', Personality.OPTIMISTIC),('SensorCell',     Personality.OPTIMISTIC), ('SensorCell',     Personality.CURIOUS)],

    # r4 (střed)
    [('SensorCell',     Personality.CURIOUS),   ('SensorCell',     Personality.CURIOUS),   ('StructuralCell', Personality.OPTIMISTIC), ('StructuralCell', Personality.OPTIMISTIC),
     ('MemoryCell',     Personality.OPTIMISTIC),('PlantCell',      Personality.OPTIMISTIC),('MemoryCell',     Personality.CAUTIOUS),
     ('StructuralCell', Personality.CAUTIOUS),  ('StructuralCell', Personality.OPTIMISTIC),('SensorCell',     Personality.CURIOUS),    ('SensorCell',     Personality.CAUTIOUS)],

    # r5
    [('SensorCell',     Personality.CURIOUS),   ('SensorCell',     Personality.OPTIMISTIC),('StructuralCell', Personality.OPTIMISTIC), ('StructuralCell', Personality.CAUTIOUS),
     ('MemoryCell',     Personality.CAUTIOUS),  ('PlantCell',      Personality.OPTIMISTIC),('MemoryCell',     Personality.CAUTIOUS),
     ('StructuralCell', Personality.OPTIMISTIC),('StructuralCell', Personality.OPTIMISTIC),('SensorCell',     Personality.CURIOUS),    ('SensorCell',     Personality.CURIOUS)],

    # r6
    [('SensorCell',     Personality.CURIOUS),   ('SensorCell',     Personality.CAUTIOUS),  ('StructuralCell', Personality.OPTIMISTIC), ('StructuralCell', Personality.OPTIMISTIC),
     ('MemoryCell',     Personality.CAUTIOUS),  ('MemoryCell',     Personality.CAUTIOUS),  ('MemoryCell',     Personality.CURIOUS),
     ('StructuralCell', Personality.CAUTIOUS),  ('StructuralCell', Personality.OPTIMISTIC),('SensorCell',     Personality.OPTIMISTIC), ('SensorCell',     Personality.CURIOUS)],

    # r7
    [('SensorCell',     Personality.CURIOUS),   ('SensorCell',     Personality.CURIOUS),   ('StructuralCell', Personality.OPTIMISTIC), ('StructuralCell', Personality.OPTIMISTIC),
     ('MemoryCell',     Personality.OPTIMISTIC),('MemoryCell',     Personality.CAUTIOUS),  ('MemoryCell',     Personality.CAUTIOUS),
     ('StructuralCell', Personality.OPTIMISTIC),('StructuralCell', Personality.CAUTIOUS),  ('SensorCell',     Personality.CURIOUS),    ('SensorCell',     Personality.CAUTIOUS)],

    # r8 (bottom)
    [('SensorCell',     Personality.CAUTIOUS),  ('SensorCell',     Personality.CURIOUS),   ('StructuralCell', Personality.OPTIMISTIC), ('StructuralCell', Personality.OPTIMISTIC),
     ('MemoryCell',     Personality.CAUTIOUS),  ('MemoryCell',     Personality.OPTIMISTIC),('MemoryCell',     Personality.CAUTIOUS),
     ('StructuralCell', Personality.OPTIMISTIC),('StructuralCell', Personality.CAUTIOUS),  ('SensorCell',     Personality.CURIOUS),    ('SensorCell',     Personality.OPTIMISTIC)],
]

# -----------------
# Parameters for Mirror Module (Digital Mirror / Digitální zrcadlo)
# -----------------
# Mirror Module provides phenomenological reflection of LLM outputs.
# The model "sees itself" through a mirror before presenting output to user.

# --- Main switch for Mirror Module ---
USE_MIRROR_MODULE = True  # Enable/disable the entire mirror prediction loop

# --- Lambda parameters for mirror transformation ---
# λ_deixis: Controls deictic swap intensity (JÁ ↔ TY / I ↔ YOU)
#   0.0 = no swap, 1.0 = full swap of all pronouns
MIRROR_LAMBDA_DEIXIS = 1.0

# λ_styl: Controls style transformation intensity
#   0.0 = no style change, 1.0 = full style adaptation
MIRROR_LAMBDA_STYL = 0.3

# --- Context parameters for swap ---
# Number of last tokens to include in swapped context
# (Larger = more context for prediction, but slower)
MIRROR_CONTEXT_LAST_TOKENS = 150

# Maximum tokens for generating expected response
MIRROR_EXPECTATION_MAX_TOKENS = 80

# --- Vector comparison parameters ---
# Method for comparing expected vs actual response vectors
# Options: "cosine", "l2", "combined"
MIRROR_COMPARISON_METHOD = "cosine"

# --- Prediction error thresholds ---
# These define how prediction error maps to hormone changes
# prediction_error ∈ [0.0, 2.0] for cosine distance (1 - similarity)

# Good prediction threshold (below this = model predicted well)
MIRROR_ERROR_THRESHOLD_LOW = 0.25

# Bad prediction threshold (above this = model was surprised)
MIRROR_ERROR_THRESHOLD_HIGH = 0.60

# --- Hormone modulation based on prediction error ---
# Good prediction (error < LOW): model correctly anticipated user
MIRROR_GOOD_PREDICTION = {
    "serotonin": 0.025,   # Stability from correct prediction
    "oxytocin": 0.030,    # Social bonding (understood the user)
    "kortizol": -0.015,   # Reduced stress
    "dopamin": 0.005,     # Mild satisfaction
}

# Bad prediction (error > HIGH): model was surprised by user
MIRROR_BAD_PREDICTION = {
    "serotonin": -0.010,  # Slight destabilization
    "oxytocin": -0.005,   # Reduced social confidence
    "kortizol": 0.035,    # Stress from misprediction
    "dopamin": 0.025,     # Curiosity/learning signal
}

# Neutral prediction (error between LOW and HIGH)
MIRROR_NEUTRAL_PREDICTION = {
    "serotonin": 0.005,
    "oxytocin": 0.005,
    "kortizol": 0.000,
    "dopamin": 0.010,
}

# --- Memory integration ---
# Whether to store expectation in AssociativeMemory for building user model
MIRROR_STORE_EXPECTATIONS = True

# Weight for expectation memories (0.0 - 1.0)
MIRROR_EXPECTATION_MEMORY_WEIGHT = 0.7

# --- Swap method ---
# Method for deictic swap: "text" (regex-based) or "embedding" (vector space)
# "embedding" is more sophisticated but requires swap vector derivation
MIRROR_SWAP_METHOD = "embedding"

# Path to precomputed swap vector file
SWAP_VECTOR_PATH = "swap_vector.pt"

# Scale factor for swap vector (affects intensity of transformation)
SWAP_VECTOR_SCALE = 0.5

# --- Debug/logging ---
# Print detailed mirror process information
MIRROR_VERBOSE = False

# Extended debug mode - shows full mirror outputs during generation
# When True, displays:
#   - Original model response (before showing to user)
#   - Lambda values (λ_deixis, λ_styl)
#   - Swapped context (after deictic swap)
#   - Expected user response (model's prediction of what user will say)
#   - Comparison results (prediction error, cosine similarity)
MIRROR_DEBUG = True

# Debug output formatting
MIRROR_DEBUG_SEPARATOR = "=" * 60
MIRROR_DEBUG_PREFIX = "[🪞 MIRROR DEBUG]"