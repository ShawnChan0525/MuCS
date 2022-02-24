from xml.dom import INDEX_SIZE_ERR
from torch.utils.data import Dataset
import torch
import os
import tqdm
from tqdm import tqdm, trange
import json
import numpy as np


class TrainingInstance():
    def __init__(self, input_ids, output_ids, src_mask, tgt_mask, awp_label, scp_label, code_size,NL_size):
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.src_mask = src_mask
        self.tgt_mask = tgt_mask
        self.awp_label = awp_label
        self.scp_label = scp_label
        self.code_size = code_size
        self.NL_size = NL_size


def read_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = json.loads(f.read())
        words = ['PAD'] + ['UNK'] + ['CLS'] + ['EOS'] + words
        vocab_size = len(words)
        word_to_id = dict(zip(words, range(len(words))))
    return word_to_id, vocab_size


def file_to_id(word_to_id, data):
    for i in range(len(data)):
        data[i] = word_to_id[data[i]
                             ] if data[i] in word_to_id else word_to_id['UNK']
    return data

def pad_input(arr:list, max_seq):
    if len(arr) > max_seq:
        arr = arr[:max_seq]
    if len(arr) < max_seq:
        padding = [0]*(max_seq-len(arr))
        arr.extend(padding)
    return arr

def get_mask(input_ids:list, is_ulm=False):
    ''' 
        maskT*mask再取下三角
        input_ids: list
        return: tensor shape [len(list), len(list)]
    '''
    input_ids = torch.Tensor(input_ids)
    mask = (input_ids != 0).long()
    mask = mask.unsqueeze(dim=0)
    output_mask = torch.mm(torch.transpose(mask, 0, 1), mask)
    if is_ulm:
        # 取下三角矩阵
        output_mask = torch.tril(output_mask, 0)
    return output_mask

def get_instances(code_path, NL_path, SCP_path, AWP_path, code_vocab_path, NL_vocab_path, input_len, output_len, with_ulm):
    scp_list = open(SCP_path, 'r', encoding='utf-8').readlines()
    awp_list = open(AWP_path, 'r', encoding='utf-8').readlines()
    assert(len(awp_list)==len(scp_list))
    inst_num = len(awp_list)
    code_list = open(code_path, 'r', encoding='utf-8').readlines()
    code_list = [x for x in code_list if x!='\n']
    NL_list = open(NL_path, 'r', encoding='utf-8').readlines()
    NL_list = [x for x in NL_list if x!='\n']

    code2id, vocab_szie = read_vocab(code_vocab_path) # 30000
    # print("vocab_szie= ", vocab_szie)
    NL2id, NL_vocab_size = read_vocab(NL_vocab_path) # 5680
    # print("NL_vocab_size ", NL_vocab_size)
    instances = []
    for i in trange(inst_num):
        code_line = code_list[i].strip()
        NL_line = NL_list[i].strip()
        tokens = json.loads(code_line)
        comment = json.loads(NL_line)
        input_tokens = ['CLS']
        input_tokens.extend(tokens)
        # 只放开始符，不放结束符
        # input_tokens.append('EOS')
        input_ids = file_to_id(code2id, input_tokens)
        code_size = len(input_ids)
        input_ids = pad_input(input_ids,input_len)
        src_mask = get_mask(input_ids, with_ulm)
        output_tokens = ['CLS']
        output_tokens.extend(comment) 
        # output_tokens.append('EOS')
        output_ids = file_to_id(NL2id, comment)
        NL_size = len(output_ids)
        output_ids = pad_input(output_ids,output_len)
        tgt_mask = get_mask(output_ids, with_ulm)
        instance = TrainingInstance(input_ids, output_ids, src_mask, tgt_mask, int(awp_list[i]), int(scp_list[i]),code_size,NL_size)
        instances.append(instance)

    # with open(output_path, 'w')as f:
    #     instances = json.dumps(instances, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    #     json.dump(instances, f)
    return instances


class BertDataset(Dataset):
    # def __init__(self, input_ids, output_ids, awp_label, scp_label):
    #     self.input_ids = input_ids
    #     self.output_ids = output_ids
    #     self.awp_label = awp_label
    #     self.scp_label = scp_label
    #     self.inst_nums = len(input_ids)

    def __init__(self, instances:TrainingInstance):
        self.instances = instances
        self.inst_nums = len(instances)

    def __len__(self):
        return self.inst_nums

    def __getitem__(self, index):
        instance = self.instances[index]
        output = {"input_ids": instance.input_ids,
                  "output_ids": instance.output_ids,
                  "src_mask": instance.src_mask,
                  "tgt_mask": instance.tgt_mask,
                  "awp_label": instance.awp_label,
                  "scp_label": instance.scp_label,
                  "code_size": instance.code_size}
        return {key: torch.tensor(value) for key, value in output.items()}


if __name__=='__main__':
    print(get_mask([1,2,3,4,5,6,0,0,0,0],True))
