import sys
from pathlib import Path
import torch
from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(MODEL_DIR))

from model.tokenizer import CharTokenizer
from model.model import TinyGPT

app = FastAPI(title="BuyWisely Custom LLM API")

tokenizer = torch.load(MODEL_DIR / "tokenizer.pt", weights_only=False)
model = TinyGPT(tokenizer.vocab_size)
model.load_state_dict(torch.load(MODEL_DIR / "buywisely_tiny_llm.pt", weights_only=True))
model.eval()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

@app.get("/health")
def health():
    return {"status": "Custom LLM server running"}

@app.post("/generate")
def generate(request: GenerateRequest):
    context = request.prompt[-128:]
    ids = torch.tensor([tokenizer.encode(context)])

    generated = ids

    with torch.no_grad():
        for _ in range(request.max_tokens):
            logits = model(generated[:, -128:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)

    response_text = tokenizer.decode(generated[0].tolist())

    return {
        "response": response_text,
        "model": "BuyWisely-TinyGPT"
    }