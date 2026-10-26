class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list):
        return "".join(self.itos[i] for i in ids)