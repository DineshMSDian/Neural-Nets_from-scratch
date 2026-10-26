import torch

# load dataset
with open(r'makemore\dataset.txt', 'r') as f:
    words = f.read().splitlines()

# cleaning
words = [word.strip().lower().replace('â€™', '') for word in words]

# bigram implementation : bigram means looking the two consecutive characters at a time
def count_bigram_in_dict(data):
    b = {}
    for w in data:
        chs = ['<s>'] + list(w) + ['<E>']
        for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            b[bigram] = b.get(bigram, 0) + 1
    return b




if __name__ == '__main__':
    print(words[:2])
    b = count_bigram_in_dict(words[:2])
    print(b)