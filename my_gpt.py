import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import BPE

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps'
#print(torch.backends.mps.is_available())

block_size = 128
batch_size = 32
eval_interval = 200
epochs = 5_000
n_embd = 64
n_head = 8
n_layer = 6
dropout = 0.3

LOAD_WEIGHTS = True
TRAIN = False


with open('data/dataset.txt', 'r') as f:
  text = f.read()


print(f"Dataset contains {len(text)} characters")
#vocab_size = len(chars)

#encoder = CharLevelEncoder()
encoder = BPE()
encoder.train(num_merges=100)
vocab_size = len(encoder.vocab)
print('vocab_size: ', vocab_size)
# Create tensors from data
data = torch.tensor(encoder.encode(text), dtype=torch.long, device=device)

# Split train and validation sets
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]

torch.manual_seed(42)


def get_batch(split='train'):
  data = train_data if split=='train' else val_data
  idx = torch.randint(len(data) - block_size, (batch_size,))
  X = torch.stack([data[i:i+block_size] for i in idx])
  y = torch.stack([data[i+1:i+1+block_size] for i in idx])
  X, y = X.to(device), y.to(device)
  return X, y

class Head(nn.Module):
  """Single head of self attention"""
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    # Compute affinities
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  """Multiple heads of self-attention in parallel """
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out
  
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa_heads = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class Transformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
    self.position_embedding_table = nn.Embedding(block_size, n_embd, device=device)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, X, targets=None):
    B, T = X.shape
    tok_emb = self.token_embedding_table(X)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln(x)
    logits = self.lm_head(x)
    if targets is None:
      loss = None
      return logits, loss
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  def generate(self, X, max_tokens=100):
    for _ in range(max_tokens):
      X_cond = X[:, -block_size:]
      logits, loss = self(X_cond)
      logits = logits[:,-1,:]
      probs = F.softmax(logits, dim=1)
      X_next = torch.multinomial(probs, num_samples=1)
      X = torch.cat([X, X_next], dim=1)
    return X
  
model = Transformer().to(device)

if LOAD_WEIGHTS:
  model.load_state_dict(torch.load('weights.pth', weights_only=True, map_location=torch.device(device)))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

def get_model_generation(max_tokens=100):
  start = torch.zeros((1,1), dtype=torch.long, device=device)
  out = model.generate(start, max_tokens)
  return encoder.decode(out.squeeze().detach().tolist())

@torch.inference_mode()
def estimate_loss():
  model.eval()
  out = {}
  for split in ['train', 'val']:
    losses = torch.zeros(eval_interval)
    for i in range(eval_interval):
      Xb, yb = get_batch(split)
      _, loss = model(Xb, yb)
      losses[i] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

if TRAIN:
  for epoch in range(epochs):
    Xb, yb = get_batch(split='train')
    logits, loss = model(Xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if epoch % eval_interval == 0:
      eval_loss = estimate_loss()
      print(f"Epoch: {epoch} | Train loss: {eval_loss['train']:.4} | Val loss: {eval_loss['val']:.4}")

print(get_model_generation(max_tokens=500))