# encoding: utf-8
import os
import json
import random
from tqdm import tqdm
from java_tokenizer import *
from collections import Counter
import six
import traceback

extra_token_num = 4


def get_files(projects, project_dict):
    '''将所有project中的所有files合在一起并打乱'''
    files = []
    for project in projects:
        files.extend(project_dict[project])
    random.seed(5)
    random.shuffle(files)
    return files

# files = get_files(test_projects, project_dict)


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        # elif isinstance(text, unicode):
        #     return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def tokenize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens, _, _, lineno = tokenize_java(text)
    lines = []
    pre_line = lineno[0]
    cur_line = []
    for i in range(len(lineno)):
        if lineno[i] == pre_line:
            cur_line.append(tokens[i])
        else:
            if cur_line == ['}']:
                lines[-1].append(cur_line[0])
            else:
                lines.append(cur_line)
            cur_line = []
            pre_line = lineno[i]
            cur_line.append(tokens[i])
        if i == len(lineno) - 1:
            if cur_line == ['}']:
                lines[-1].append(cur_line[0])
            else:
                lines.append(cur_line)
    return lines


def build_vocab(data, vocab_size=None, vocab_path=None):
    words = []
    for line in tqdm(data):
        words.extend(line)
    counter = Counter(words)
    if vocab_size:
        counter_pairs = counter.most_common(vocab_size - extra_token_num)
    else:
        counter_pairs = counter.most_common()
    words, values = list(zip(*counter_pairs))

    words = list(words)
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(words))



def create_training_corpus(project_path, file_num):
    '''获取code和comment的tokens文件，以及vocabs文件'''
    code_dest = open(project_path+"/tokens.txt", 'w', encoding='utf-8')
    comment_dest = open(project_path+"/comment_tokens.txt", 'w', encoding='utf-8')
    documents = [] # documents实际上就是一个数组，里面放着code和comment中所有的tokens
    comment_tokens= [] # 将comment中的tokens专门放到一个文件里
    for file in tqdm(range(file_num)):
        try:
            code = tokenize_file(os.path.join(project_path, "code", str(file)+'.txt'))
            comment = tokenize_file(os.path.join(project_path, "comment", str(file)+'.txt'))
            documents.extend(code)
            comment_tokens.extend(comment)
            for line in code:
                code_dest.write(json.dumps(line) + '\n')
            for line in comment:
                comment_dest.write(json.dumps(line) + '\n')
        except Exception as e:
            # print("wrong!! "+ str(file))
            # print("Wrong type: "+ e.__class__.__name__)
            # print("Wrong detail: "+ str(e))
            # traceback.print_exc()
            pass
    code_dest.close()
    comment_dest.close()
    build_vocab(documents, 30000, project_path + "/vocabs.txt")
    build_vocab(comment_tokens, 30000, project_path + "/comment_vocabs.txt")
    return documents,comment_tokens


def save_vocab(data_path, size, path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        tokens = []
        for line in tqdm(data):
            line = line.strip()
            if line:
                tokens.append(json.loads(line))
        # print(tokens[:100])
        build_vocab(tokens, vocab_size=size, vocab_path=path)

def getSCP():
    input_data  = open("data/raw_data/label.txt",'r',encoding="utf-8")
    output_data = open("data/processed_data/SCP_labels.txt",'w',encoding="utf-8")
    SCP_list = ['what\n','why\n','how_to_use\n','how_it_is_done\n','property\n','others\n']
    labels = input_data.readlines()
    for label in tqdm(labels):
        output_data.write(str(SCP_list.index(label))+'\n')

if __name__ == "__main__":
    token_file = 'data/processed_data/tokens.txt'
    # token_corpus = FLAGS.token_file # tokens.txt
    project_path = 'C:/Users/Shawnchan/Desktop/iSE/Multi-task code summerization/MuCS/data/data'
    documents = create_training_corpus(project_path,20000)
    # getSCP()
