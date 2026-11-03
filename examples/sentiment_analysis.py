import torch, torch.nn as nn
import random , re
torch.manual_seed(0); random.seed(0)

templates = ["the {n} was {a}", "a truly {a} {n}", "i thought the {n} was {a}", "{a} {n} overall", "what a {a} {n}"]
pos_w = ["great", "amazing", "wonderful", "loved", "brilliant", "fun", "beautiful", "excellent"]
neg_w = ["terrible", "awful", "boring", "hated", "dull", "bad", "worst", "weak"]
nouns = ["movie", "film", "story", "acting", "plot", "ending", "soundtrack"]

def make(n):
    data = []
    for _ in range(n):
        label = random.randint(0, 1)
        adj = random.choice(pos_w if label else neg_w)
        data.append((random.choice(templates).format(n=random.choice(nouns), a=adj), label))
    return data
train, test = make(2000), make(400)

tok = lambda s: re.findall(r"[a-z']+", s.lower())
vocab = ["<unk>"] + sorted({w for s, _ in train for w in tok(s)})
stoi = {w: i for i, w in enumerate(vocab)}
enc = lambda s: torch.tensor([stoi.get(w, 0) for w in tok(s)])

class SentimentNet(nn.Module):
    def __init__(self, V, d=32):
        super().__init__()
        self.emb = nn.EmbeddingBag(V, d, mode="mean")   # embed + average in one op
        self.fc = nn.Linear(d, 2)
    def forward(self, ids, offsets): return self.fc(self.emb(ids, offsets))

def batchify(rows):
    ids = [enc(s) for s, _ in rows]
    offsets = torch.tensor([0] + [len(x) for x in ids[:-1]]).cumsum(0)  # where each sentence starts
    return torch.cat(ids), offsets, torch.tensor([y for _, y in rows])

model = SentimentNet(len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(5):
    random.shuffle(train)
    for i in range(0, len(train), 64):
        ids, off, y = batchify(train[i:i + 64])
        loss = nn.functional.cross_entropy(model(ids, off), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ids, off, y = batchify(test)
        acc = (model(ids, off).argmax(1) == y).float().mean()
    print(f"epoch {epoch}  loss {loss.item():.3f}  test acc {acc:.1%}")

def predict(text):
    with torch.no_grad():
        p = model(enc(text), torch.tensor([0])).softmax(-1)[0, 1].item()
        return f"{'👍' if p > .5 else '👎'} {p:.2f}  '{text}'"


# 🔍 peek inside: score each word on its own
with torch.no_grad():
    w_scores = model.fc(model.emb.weight)[:, 1] - model.fc(model.emb.weight)[:, 0]
top = w_scores.argsort()
print("most 👎:", [vocab[i] for i in top[:4]])
print("most 👍:", [vocab[i] for i in top[-4:]])