import torch

def load():
    # load dataset
    with open(r'makemore\dataset.txt', 'r') as f:
        words = f.read().splitlines()

    # cleaning
    words = [word.strip().lower().replace('â€™', '') for word in words]
    return words

# bigram implementation : bigram means looking the two consecutive characters at a time

def count_bigram_in_dict(words):
    b = {}
    for w in words:
        chs = ['<s>'] + list(w) + ['<E>']
        for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            b[bigram] = b.get(bigram, 0) + 1
    
    return b

def count_bigram_in_torch(words):
    N = torch.zeros((28, 28), dtype=torch.int32) # created a 2d array od 28x28
    chrs = sorted(set(''.join(words)))
    stoi = {s:i for i,s in enumerate(chrs)} # lookup table
    stoi['<S>'] = 26
    stoi['<E>'] = 27

    for w in words:
        chs = ['<S>'] + list(w) + ['<E>']

        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
    return N
            



if __name__ == '__main__':

    words = load()
    print(words[:2])

    b = count_bigram_in_dict(words[:10])
    print(sorted(b.items(), key = lambda kv: -kv[1]))
    print('-'*20)
    N = count_bigram_in_torch(words[:10])
    print(N)