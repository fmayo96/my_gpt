"""
This is a tokenizer that generates tokens for a given dataset using the byte pairing encoding.
"""

class BPE():
  def __init__(self):
    self.tokens = []
    self.merges = {}
    self.vocab = {}
  
  def get_stats(self, ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
      counts[pair] = counts.get(pair, 0) + 1
    return counts

  def merge(self, ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
      if i < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
        new_ids.append(idx)
        i += 2
      else:
        new_ids.append(ids[i])
        i += 1
    return new_ids   

  def create_vocab(self, num_merges):
    ids = list(self.tokens)
    for i in range(num_merges):
      stats = self.get_stats(ids)
      pair = max(stats, key=stats.get)
      idx = 256 + i
      ids = self.merge(ids, pair, idx)
      self.merges[pair] = idx
    return ids

  def train(self, num_merges):
    with open('data/dataset.txt', 'r') as f:
      text = f.read()
    self.tokens = text.encode('utf8')
    self.tokens = list(map(int, self.tokens))
    self.tokens = self.create_vocab(num_merges)
    self.vocab = {idx: bytes([idx]) for idx in range(256)}

    for (p0, p1), idx in self.merges.items():
      self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

  def decode(self, ids):
    tokens = b"".join(self.vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors='replace')
    return text

  def encode(self, text):
    tokens = list(text.encode('utf-8'))
    while len(tokens) >= 2:
      stats = self.get_stats(tokens)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break
      idx = self.merges[pair]
      tokens = self.merge(tokens, pair, idx)
    return tokens

