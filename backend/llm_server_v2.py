import sys
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.byte_tokenizer import ByteTokenizer
from model.model import TinyGPT


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "model"
    / "artifacts_v2"
    / "best_checkpoint.pt"
)

DEVICE = torch.device("cpu")
generation_lock = Lock()

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"V2 checkpoint not found: {CHECKPOINT_PATH}. "
        "Run model/train_v2.py first."
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)

if checkpoint.get("format_version") != 2:
    raise ValueError("Expected a version 2 training checkpoint.")

tokenizer = ByteTokenizer()

if checkpoint["tokenizer_spec"] != tokenizer.specification():
    raise ValueError(
        "Checkpoint tokenizer specification does not match "
        "the current ByteTokenizer."
    )

config = checkpoint["model_config"]

if config["vocab_size"] != tokenizer.vocab_size:
    raise ValueError("Tokenizer and model vocabulary sizes differ.")

# Recreate the architecture saved during training.
model = TinyGPT(**config)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

CONTEXT_LENGTH = model.block_size

app = FastAPI(title="BuyWisely V2 Custom LLM API")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=50000)
    max_tokens: int = Field(default=150, ge=1, le=512)


@app.get("/health")
def health():
    return {
        "status": "Custom LLM server running",
        "model": "BuyWisely-TinyGPT-v2",
        "device": str(DEVICE),
        "context_length": CONTEXT_LENGTH,
        "context_unit": "UTF-8 byte tokens",
        "saved_training_step": checkpoint["step"],
        "validation_loss": checkpoint["validation_loss"],
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    if not request.prompt.strip():
        raise HTTPException(
            status_code=422,
            detail="Prompt must contain non-whitespace text.",
        )

    encoded = tokenizer.encode(request.prompt)

    # Reject oversized input instead of silently discarding evidence.
    if len(encoded) > CONTEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Prompt exceeds the model context length. "
                    "Shorten the input or use a model trained "
                    "with a longer context."
                ),
                "input_tokens": len(encoded),
                "context_length": CONTEXT_LENGTH,
                "context_unit": "UTF-8 byte tokens",
            },
        )

    generated = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE,
    )

    prompt_length = generated.shape[1]

    with generation_lock:
        with torch.inference_mode():
            for _ in range(request.max_tokens):
                model_input = generated[:, -CONTEXT_LENGTH:]
                logits = model(model_input)

                next_token = torch.argmax(
                    logits[:, -1, :],
                    dim=-1,
                    keepdim=True,
                )

                generated = torch.cat(
                    [generated, next_token],
                    dim=1,
                )

    new_token_ids = generated[0, prompt_length:].tolist()

    return {
        "response": tokenizer.decode(new_token_ids),
        "model": "BuyWisely-TinyGPT-v2",
        "input_tokens": len(encoded),
        "generated_tokens": len(new_token_ids),
        "context_length": CONTEXT_LENGTH,
        "input_truncated": False,
        "warnings": [
            "Generation uses a sliding context window.",
            "No end-of-answer token is configured; generation "
            "runs for the requested number of byte tokens.",
            "Generated text may contain malformed UTF-8 sequences, "
            "displayed as replacement characters.",
            "Answer accuracy has not been established.",
        ],
    }