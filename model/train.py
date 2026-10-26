import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizer import CharTokenizer
from model import TinyGPT

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

text = (DATA_DIR / "train.txt").read_text(encoding="utf-8")

tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

block_size = 128
batch_size = 16
epochs = 10

class ShoppingDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, index):
        x = self.data[index:index + self.block_size]
        y = self.data[index + 1:index + self.block_size + 1]
        return x, y

dataset = ShoppingDataset(data, block_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TinyGPT(tokenizer.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_function = nn.CrossEntropyLoss()

for epoch in range(epochs):
    total_loss = 0

    for x, y in loader:
        logits = model(x)

        loss = loss_function(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")

torch.save(model.state_dict(), "buywisely_tiny_llm.pt")
torch.save(tokenizer, "tokenizer.pt")

print("Model training complete.")