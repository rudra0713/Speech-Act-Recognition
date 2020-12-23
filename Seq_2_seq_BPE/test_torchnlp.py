from torchnlp.word_to_vector import FastText  # doctest: +SKIP
from torchnlp.word_to_vector import BPEmb  # doctest: +SKIP
import torch
import torch.nn as nn
import torch.nn.functional as F

vectors = BPEmb(dim=300)  # doctest: +SKIP
# print(vectors['hello']) # doctest: +SKIP
# print("len ", len(vectors['hello']))


emb = nn.Embedding(20, 10)
print(emb.weight.data)
emb.weight.data = torch.eye(30)
# .weight.requires_grad=False
print(emb.weight.data)

