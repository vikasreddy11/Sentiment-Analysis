import torch 
from collections import Counter

#Tokenizer 
def tokenizer(text):
    return text.lower().split()

#Vocabulary
def build_vocab(dataset,min_frq=3):

    counter=Counter()
    for sample in dataset:
        counter.update(tokenizer(sample['text']))

    vocab={'<pad':0,'<unk>':1}
    for word,count in counter.items():
        if count>=3:
            vocab[word]=len(vocab)
        
    return vocab



