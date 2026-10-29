import sys
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"

sys.path.insert(0, str(PROJECT_ROOT))

# Compatibility with older tokenizer checkpoints saved when
# tokenizer.py was imported as a top-level module.
sys.path.append(str(MODEL_DIR))

from model.model import TinyGPT


app = FastAPI(title="BuyWisely Custom LLM API")

DEVICE = torch.device("cpu")
generation_lock = Lock()

TOKENIZER_PATH = MODEL_DIR / "tokenizer.pt"
WEIGHTS_PATH = MODEL_DIR / "buywisely_tiny_llm.pt"

if not TOKENIZER_PATH.is_file():
    raise FileNotFoundError(
        f"Tokenizer checkpoint not found: {TOKENIZER_PATH}"
    )

if not WEIGHTS_PATH.is_file():
    raise FileNotFoundError(
        f"Model checkpoint not found: {WEIGHTS_PATH}"
    )

# Only load this pickle-based tokenizer if you created it
# and trust the file.
tokenizer = torch.load(
    TOKENIZER_PATH,
    map_location="cpu",
    weights_only=False,
)

# These defaults must match the architecture used in training.
model = TinyGPT(tokenizer.vocab_size)

state_dict = torch.load(
    WEIGHTS_PATH,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

CONTEXT_LENGTH = model.block_size


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=50000)
    max_tokens: int = Field(default=150, ge=1, le=512)


@app.get("/health")
def health():
    return {
        "status": "Custom LLM server running",
        "model": "BuyWisely-TinyGPT",
        "device": str(DEVICE),
        "context_length": CONTEXT_LENGTH,
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    if not request.prompt.strip():
        raise HTTPException(
            status_code=422,
            detail="Prompt must contain non-whitespace text.",
        )

    # The earlier CharTokenizer exposes its vocabulary as stoi.
    unsupported_characters = sorted({
        character
        for character in request.prompt
        if character not in tokenizer.stoi
    })

    encoded = tokenizer.encode(request.prompt)

    if not encoded:
        raise HTTPException(
            status_code=422,
            detail=(
                "The tokenizer could not encode the prompt. "
                "Check the tokenizer vocabulary."
            ),
        )

    input_token_count = len(encoded)
    input_truncated = input_token_count > CONTEXT_LENGTH

    # Keep the final context window, but disclose this explicitly.
    context_ids = encoded[-CONTEXT_LENGTH:]

    generated = torch.tensor(
        [context_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    prompt_length = generated.shape[1]

    # Serialize generation for this simple single-process server.
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

    # Return only newly generated tokens, not the original prompt.
    new_token_ids = generated[0, prompt_length:].tolist()
    response_text = tokenizer.decode(new_token_ids)

    warnings = []

    if input_truncated:
        warnings.append(
            "The prompt exceeded the model context length. "
            "Only the final context window was used."
        )

    if unsupported_characters:
        warnings.append(
            "The current tokenizer dropped unsupported characters."
        )

    return {
        "response": response_text,
        "model": "BuyWisely-TinyGPT",
        "context_length": CONTEXT_LENGTH,
        "input_tokens": input_token_count,
        "used_input_tokens": len(context_ids),
        "generated_tokens": len(new_token_ids),
        "input_truncated": input_truncated,
        "unsupported_characters": unsupported_characters,
        "warnings": warnings,
    }