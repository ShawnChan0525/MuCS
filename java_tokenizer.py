# encoding: utf-8
import javalang


def tokenize_java(code, code_lines=None, need_type_info=False, need_index=False):
    token_gen = javalang.tokenizer.tokenize(code)
    tokens = []
    indexes = []
    while (True):
        try:
            token = next(token_gen)
        except:
            break
        tokens.append(token)

    pure_tokens = [token.value for token in tokens]

    pos = [[token.position[0], token.position[1]] for token in tokens]
    lineno = [token.position[0] for token in tokens]

    if need_index:
        indexes = []
        for i in range(len(pure_tokens)):
            start_index = ''.join(code_lines[:(pos[i][0]-1)])
            start_index = len(start_index)+pos[i][1]-1
            end_index = start_index + len(pure_tokens[i])
            indexes.append([start_index, end_index])

    if need_type_info:
        token_types = [str(type(token))[:-2].split(".")[-1] for token in tokens]
        return pure_tokens, token_types, pos, indexes, lineno
    else:
        return pure_tokens, pos, indexes, lineno

def get_type_corpus(tokens, types):
    for i in range(len(types)):
        if types[i] != 'Identifier':
            types[i] ='_'
        else:
            types[i] = tokens[i]
    return types

if __name__ == "__main__":
    with open('data/demo.java', 'r') as f:
        text = f.read()
    with open('data/demo.java', 'r') as f:
        text_lines = f.readlines()
    # text = FuncExample
    # tokens, types, pos, indexes, lineno = tokenize_java(text, text_lines, True, True)
    tokens, types, _, _, _ = tokenize_java(text, text_lines, True, True)
    types = get_type_corpus(tokens,types)
    for i in range(len(tokens)):
        print(tokens[i], end=',')
        print(types[i])
        # print(pos[i], end=',')
        # print(indexes[i], end=',')
        # print(lineno[i])

