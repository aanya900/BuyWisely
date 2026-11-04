import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from byte_tokenizer import ByteTokenizer
from model import TinyGPT


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT.parent / "docs" / "learning-diagnostic.json"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = ByteTokenizer()

PROMPT = (
    "Product: Earbuds\n"
    "Battery: 30 hours\n"
    "Question: What battery life is listed?\n"
    "Answer:"
)

EXPECTED_COMPLETION = " The listed battery life is 30 hours."

training_text = PROMPT + EXPECTED_COMPLETION

encoded = tokenizer.encode(training_text)
prompt_ids = tokenizer.encode(PROMPT)

CONFIG = {
    "vocab_size": tokenizer.vocab_size,
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 2,
    "block_size": 256,
}

if len(encoded) > CONFIG["block_size"]:
    raise ValueError("Diagnostic example exceeds the context length.")

x = torch.tensor(
    [encoded[:-1]],
    dtype=torch.long,
    device=DEVICE,
)

y = torch.tensor(
    [encoded[1:]],
    dtype=torch.long,
    device=DEVICE,
)

model = TinyGPT(**CONFIG).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
)

MAX_STEPS = 1000
CHECK_INTERVAL = 50
history = []


@torch.no_grad()
def generate_completion():
    model.eval()

    generated = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    start_length = generated.shape[1]

    # Known length is used only for this memorization diagnostic.
    for _ in range(len(tokenizer.encode(EXPECTED_COMPLETION))):
        logits = model(generated[:, -model.block_size:])

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    result = tokenizer.decode(
        generated[0, start_length:].tolist()
    )

    model.train()
    return result


print(f"Device: {DEVICE}")
print("Diagnostic: memorizing one supplied training example.")
print("This is not an evaluation on unseen data.")

model.train()

memorized = False
completion = ""
step = 0

for step in range(1, MAX_STEPS + 1):
    optimizer.zero_grad(set_to_none=True)

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    if step == 1 or step % CHECK_INTERVAL == 0:
        completion = generate_completion()
        memorized = completion == EXPECTED_COMPLETION

        history.append({
            "step": step,
            "training_loss": loss.item(),
            "completion": completion,
            "exact_training_example_match": memorized,
        })

        print(f"\nStep {step} | Loss: {loss.item():.4f}")
        print(f"Completion: {completion!r}")
        print(f"Exact match: {memorized}")

        if memorized:
            break

report = {
    "experiment": "single_example_memorization",
    "device": str(DEVICE),
    "model_config": CONFIG,
    "prompt": PROMPT,
    "expected_completion": EXPECTED_COMPLETION,
    "actual_completion": completion,
    "exact_training_example_match": memorized,
    "steps_completed": step,
    "history": history,
    "limitations": [
        "The prompt and answer were included in training.",
        "The output length was supplied to the diagnostic.",
        "Success does not demonstrate generalization.",
        "No V2 checkpoint was loaded or overwritten.",
    ],
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

REPORT_PATH.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\nReport saved: {REPORT_PATH}")

if memorized:
    print("PASS: The model memorized this training example.")
else:
    print(
        "NOT RESOLVED: The example was not reproduced exactly "
        "within the configured training budget."
    )