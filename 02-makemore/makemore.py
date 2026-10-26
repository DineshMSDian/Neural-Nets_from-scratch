import torch
import matplotlib.pyplot as plt

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
    N = torch.zeros((27, 27), dtype=torch.int32) # created a 2d array od 28x28
    chrs = sorted(set(''.join(words)))
    stoi = {s:i+1 for i,s in enumerate(chrs)} # lookup table
    stoi['.'] = 0

    for w in words:
        chs = ['.'] + list(w) + ['.']

        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
    return N, stoi
            
def visualize(N, stoi):
    itos = {i:s for s, i in stoi.items()}
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
            plt.text(j, i, N[i,j].item(), ha='center', va='top', color='gray')

    plt.axis('off')
    plt.show()


if __name__ == '__main__':

    words = load()
    print(words[:2])

    b = count_bigram_in_dict(words[:10])
    print(sorted(b.items(), key = lambda kv: -kv[1]))
    print('-'*20)
    N, stoi = count_bigram_in_torch(words)
    print(N)

    visualize(N, stoi)