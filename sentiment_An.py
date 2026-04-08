import torch
from collections import Counter

#Tokenizer 
def tokenizer(text):
    return text.lower().split()

#Vocabulary
def vocab(datasets,min_frq=3):

    counter=Counter()
    for sample in datasets:
        counter.update(tokenizer(sample('text')))

    vocab={'PAD':0,'UNK':1}
    for word,count in counter.items():
        if count>=min_frq:
            vocab[word]=len(word)
    return vocab

#Imdb dataset
class IMDBdataset(torch.utils.data.Dataset):
    def __init__(self,data,vocab):
        self.data=[]
        self.vocab=vocab
        unk=vocab['UNK']

        for sample in data:
            tokens=tokenizer(sample)
            indices=[vocab.get(t,unk) for t in tokens]
            labels=sample['label']
            self.data.append(torch.tensor((indices,dtype=t)))