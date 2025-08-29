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
pretrain_num_epochs = 55       # Adjust according to the situation
pretrain_learning_rate = 3e-4  # Max LR
pretrain_min_lr = 1e-5         # Min LR on the end
pretrain_warmup_iters = 900    # warmup steps, very small model - 2000 is better
pretrain_batch_size = 6        # by your HW
pretrain_max_seq_len = 512     # very small model - 4096 is better

# -----------------
# Parameters for fine-tuning (finetune.py)
# -----------------
finetune_data_path = 'data/CZ_QA_MIKRO.txt'   #replace with yours, see example
base_model_path = 'checkpoints/base_model/latest_checkpoint.pt'
finetune_checkpoint_dir = 'checkpoints/finetuned_model'
finetune_num_epochs = 90       # Adjust according to the situation
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
USE_PLANT_NETWORK = False # always False during pre-training and fine-tuning.
# -------------------------------------------------------------------------

# File for saving and loading the state of the plant network.
PLANT_NET_STATE_FILE = "plant_net_state.json"

# Definition of personalities (Enum)
class Personality(Enum):
    OPTIMISTIC = 1
    CAUTIOUS = 2
    CURIOUS = 3

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