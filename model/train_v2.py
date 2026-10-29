import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from byte_tokenizer import ByteTokenizer
from model import TinyGPT


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts_v2"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# These are starter settings, not tuned final settings.
CONFIG = {
    "vocab_size": 256,
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 2,
    "block_size": 512,
}

BATCH_SIZE = 2
TRAIN_STEPS = 200
EVAL_INTERVAL = 50
EVAL_BATCHES = 10
LEARNING_RATE = 3e-4

tokenizer = ByteTokenizer()

train_path = DATA_DIR / "train.txt"
validation_path = DATA_DIR / "validation.txt"

for path in (train_path, validation_path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")

train_text = train_path.read_text(encoding="utf-8")
validation_text = validation_path.read_text(encoding="utf-8")

train_data = torch.tensor(
    tokenizer.encode(train_text),
    dtype=torch.long,
)

validation_data = torch.tensor(
    tokenizer.encode(validation_text),
    dtype=torch.long,
)

context_length = CONFIG["block_size"]

for name, data in (
    ("training", train_data),
    ("validation", validation_data),
):
    if len(data) <= context_length:
        raise ValueError(
            f"The {name} dataset must contain more than "
            f"{context_length} byte tokens."
        )

model = TinyGPT(**CONFIG).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

# A dedicated generator makes validation batches repeatable.
def make_batch(data, generator=None):
    starts = torch.randint(
        high=len(data) - context_length,
        size=(BATCH_SIZE,),
        generator=generator,
    ).tolist()

    x = torch.stack([
        data[start:start + context_length]
        for start in starts
    ])

    y = torch.stack([
        data[start + 1:start + context_length + 1]
        for start in starts
    ])

    return x.to(DEVICE), y.to(DEVICE)


def calculate_loss(x, y):
    logits = model(x)

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )


@torch.no_grad()
def validation_loss():
    model.eval()

    generator = torch.Generator().manual_seed(SEED + 1)
    values = []

    for _ in range(EVAL_BATCHES):
        x, y = make_batch(validation_data, generator)
        values.append(calculate_loss(x, y).item())

    model.train()
    return sum(values) / len(values)


tokenizer.save(ARTIFACT_DIR / "tokenizer.json")

best_validation_loss = float("inf")
history = []
start_time = time.perf_counter()

print(f"Device: {DEVICE}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training tokens: {len(train_data):,}")
print(f"Validation tokens: {len(validation_data):,}")
print(f"Context length: {context_length} byte tokens")

model.train()

for step in range(1, TRAIN_STEPS + 1):
    x, y = make_batch(train_data)

    optimizer.zero_grad(set_to_none=True)

    loss = calculate_loss(x, y)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    if step == 1 or step % EVAL_INTERVAL == 0:
        val_loss = validation_loss()

        record = {
            "step": step,
            "train_batch_loss": loss.item(),
            "validation_loss": val_loss,
            "elapsed_seconds": round(
                time.perf_counter() - start_time,
                2,
            ),
        }

        history.append(record)

        print(
            f"Step {step}/{TRAIN_STEPS} | "
            f"train batch loss: {loss.item():.4f} | "
            f"validation loss: {val_loss:.4f}"
        )

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss

            torch.save(
                {
                    "format_version": 2,
                    "model_config": CONFIG,
                    "tokenizer_spec": tokenizer.specification(),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "validation_loss": val_loss,
                    "seed": SEED,
                },
                ARTIFACT_DIR / "best_checkpoint.pt",
            )

        (ARTIFACT_DIR / "training_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

print("Training run finished.")
print(f"Best validation loss: {best_validation_loss:.4f}")
print(f"Artifacts: {ARTIFACT_DIR}")