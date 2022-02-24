from torch.nn import Transformer,LayerNorm
import torch
import torch.nn as nn
from bert_dataset import get_instances
from models.bert_model import LayerNorm
import random
import os

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
    lsm = nn.LogSoftmax(dim=-1)
    src = torch.randn(8,6,128) # batch_size; classes;seq_len
    print(src)
    l= lsm(src)
    a = torch.argmax(l,dim=1)
    print(a)
    tgt = torch.ones(8,128).long()
    loss_fn = nn.NLLLoss()
    output = loss_fn(a,tgt)
    print(output)

def test_logsoftmax():
    lsm = nn.Softmax(dim=1)
    src = torch.randn(3,4)
    print(src)
    mid = lsm(src)
    print(mid.data)
    output = torch.argmax(mid,dim=1)
    print(output)
def test_saveandload():
    dir_demo = "data/demo"
    dir_data = "data/data"
    my_dir = dir_data
    code_path = os.path.join(my_dir, "tokens.txt")
    NL_path = os.path.join(my_dir, "comment_tokens.txt")
    SCP_path = os.path.join(my_dir, "SCP.txt")
    AWP_path = os.path.join(my_dir, "AWP.txt")
    code_vocab_path = os.path.join(my_dir, "vocabs.txt")
    NL_vocab_path = os.path.join(my_dir, "comment_vocabs.txt")
    # output_path = os.path.join(my_dir,"instances.txt")
    instances = get_instances(code_path, NL_path, SCP_path, AWP_path,
                                                         code_vocab_path, NL_vocab_path, 256, 128, True)
    torch.save(instances,os.path.join(my_dir, "instance.pth"))

def test_sum():
    a = torch.tensor([1,2,3,4,5])
    print(a.sum().item())
if __name__=="__main__":
    test_sum()