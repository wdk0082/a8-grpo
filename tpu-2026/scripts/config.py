"""Single source of truth for hyperparameters and paths.

Everything in this file is a knob you might tune. Other scripts import from here
so a change in one place propagates everywhere.
"""
import os
import jax

# ====== Model ======
MODEL_ID = "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = 0.9
DATA_SOURCE = os.environ.get("DATA_SOURCE", "tfds")  # "tfds" or "kaggle"

# ====== LoRA (parameter-efficient finetuning) ======
# Only the LoRA adapters are trained; the base model is frozen and shared with
# the reference model. Smaller rank => fewer trainable params, smaller KL drift.
RANK = 64
ALPHA = 64.0

# ====== Sharding (TPU mesh) ======
NUM_TPUS = len(jax.devices())
if NUM_TPUS == 8:
    MESH_COUNTS = (1, 4)
elif NUM_TPUS == 4:
    MESH_COUNTS = (1, 4)
elif NUM_TPUS == 1:
    MESH_COUNTS = (1, 1)
else:
    raise ValueError(f"Unsupported number of TPUs: {NUM_TPUS}")

MESH = [MESH_COUNTS, ("fsdp", "tp")]

# ====== Generation during GRPO rollouts ======
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
TEMPERATURE = 0.9          # high enough that the G samples actually differ
TOP_P = 1.0
TOP_K = 50
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", 2))  # G — group size for advantage norm (R3: 8)

# ====== GRPO loss ======
NUM_ITERATIONS = 1         # mu — PPO-style inner optimisation passes per batch
BETA = float(os.environ.get("BETA", 0.08))       # KL penalty coeff (R1: 0.3, R4: 0.0)
EPSILON = float(os.environ.get("EPSILON", 0.2))  # PPO-style clip range
ADVANTAGE_ESTIMATOR = os.environ.get("ADVANTAGE_ESTIMATOR", "grpo")  # R3: "rloo" (leave-one-out)

# ====== Reward shaping (R2: length penalty; OFF by default so R0 == baseline) ======
LENGTH_PENALTY = os.environ.get("LENGTH_PENALTY", "0") == "1"
LEN_SOFT_CHARS = int(os.environ.get("LEN_SOFT_CHARS", 1500))  # start penalising (char proxy)
LEN_MAX_CHARS = int(os.environ.get("LEN_MAX_CHARS", 3000))    # full -1 (~768 tok @ ~4 ch/tok)

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = 1
NUM_BATCHES = 3738
NUM_TEST_BATCHES = int(os.environ.get("NUM_TEST_BATCHES", 64))  # env-overridable for large-n eval (full GSM8K test = 1319)
EVAL_EVERY_N_STEPS = 64
NUM_EPOCHS = 1
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# ====== Optimiser ======
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 0.1 * MAX_STEPS
MAX_GRAD_NORM = 0.1        # tight clipping keeps KL well-behaved

# ====== Checkpointing ======
# /tmp is volatile AND shared across users on this VM → per-run persistent dir under $HOME.
RUN_NAME = os.environ.get("RUN_NAME", "default")
RUN_ROOT = os.environ.get("RUN_ROOT", os.path.expanduser(f"~/grpo_runs/{RUN_NAME}"))
INTERMEDIATE_CKPT_DIR = os.path.join(RUN_ROOT, "intermediate_ckpt/")
CKPT_DIR = os.path.join(RUN_ROOT, "ckpts/")
TENSORBOARD_DIR = os.path.join(RUN_ROOT, "tensorboard/grpo")
# Dense ckpts so the best-val (~peak) survives — baseline kept only the last 4 (lost ~step-450).
SAVE_INTERVAL_STEPS = int(os.environ.get("SAVE_INTERVAL_STEPS", 100))
MAX_TO_KEEP = int(os.environ.get("MAX_TO_KEEP", 40))

# ====== Inference presets ======
GENERATION_CONFIGS = {
    "greedy":   {"temperature": None, "top_k": 1,    "top_p": None},
    "standard": {"temperature": 0.7,  "top_k": 50,   "top_p": 0.95},
    "liberal":  {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

# ====== W&B ======
# Set WANDB_RUN_ID in env to resume an existing run (e.g. "bnh9ttlt").
# Project + entity must match the existing run, otherwise wandb won't find it.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "a8-grpo")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "sichengma0514-university-of-cambridge")
WANDB_RUN_ID = os.environ.get("WANDB_RUN_ID", None)
