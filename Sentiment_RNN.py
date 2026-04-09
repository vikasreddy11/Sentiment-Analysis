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


#Dataset
class IMDBdataset(torch.utils.data.Dataset):
    def __init__(self,data,vocab):
        self.data=[]
        self.vocab=vocab
        unk=vocab['<unk>']
        for sample in data:
            tokens=tokenizer(sample['text'])
            indices=[vocab.get(t,unk) for t in tokens]
            label=sample['label']
            self.data.append((torch.tensor(indices,dtype=torch.long),label))

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
#padding 
def collate_fn(batch):
    text,label=zip(*batch)
    padded=torch.nn.utils.rnn.pad_sequence(text,batch_first=True,padding_value=0)
    lengths=torch.tensor(len(t) for t in text)
    indices=torch.tensor(label,dtype=torch.float)
    return padded,lengths,indices

#RNN

