"""
This is a tokenizer that generates character level tokens for all the characters
present in a given dataset.
"""

with open('data/dataset.txt', 'r') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create dictionaries
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}


class CharLevelEncoder():
  def __init__(self):
  # Create encoding and decoding functions
    self.encode = lambda s: [stoi[c] for c in s]
    self.decode = lambda arr: ''.join([itos[i] for i in arr])