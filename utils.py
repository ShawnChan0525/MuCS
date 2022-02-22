import json


def id2word(word_to_id, data):
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    for i in range(len(data)):
        data[i] = id_to_word[data[i]]
    return data

def file_to_id(word_to_id, data):
    for i in range(len(data)):
        data[i] = word_to_id[data[i]
                             ] if data[i] in word_to_id else word_to_id['UNK']
    return data

def read_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = json.loads(f.read())
        words = ['PAD'] + ['UNK'] + ['CLS'] + ['EOS'] + words
        vocab_size = len(words)
        word_to_id = dict(zip(words, range(len(words))))
    return word_to_id, vocab_size

if __name__ == "__main__":
    dir = "C:/Users/Shawnchan/Desktop/iSE/Multi-task code summerization/MuCS/data/data/comment_vocabs.txt"
    dicts,_ = read_vocab(dir)
    print(id2word(dicts, [1,2,3,4]))