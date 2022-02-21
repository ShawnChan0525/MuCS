from torch.nn import Transformer,LayerNorm
import torch
import torch.nn as nn
from models.bert_model import LayerNorm
import random

def test_layer_norm():
    # NLP Example
    batch, sentence_length, embedding_dim = 2, 3, 5
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    layer_norm1 = nn.LayerNorm(embedding_dim)
    layer_norm2 = LayerNorm(embedding_dim)
    # Activate module
    print(layer_norm1(embedding))
    print(layer_norm2(embedding))

def test_nelement():
    x = torch.randn((3,4,5,6))
    print(x.nelement())

def test_shuffle():
    rng = random.Random(12345)
    a = [1,2,3,4,56,9]
    print(a)
    rng.shuffle(a)
    print(a)
    rng.shuffle(a)
    print(a)

def test_zb():
    a = [1,2,3,0,54,5,3]
    a = [x for x in a if x]
    print(a)

def test_NLLloss():
    lsm = nn.LogSoftmax(dim=1)
    src = torch.randn(3,5)
    print(src)
    tgt = torch.tensor([1,0,4])
    loss_fn = nn.NLLLoss()
    output = loss_fn(lsm(src),tgt)
    print(output)

def test_logsoftmax():
    lsm = nn.Softmax(dim=1)
    src = torch.randn(3,4)
    print(src)
    mid = lsm(src)
    print(mid.data)
    output = torch.argmax(mid,dim=1)
    print(output)

if __name__=="__main__":
    test_logsoftmax()