import torch
from collections import Counter

#Tokenizer 
def tokenizer(text):
    return text.lower().split()

#Vocabulary
def build_vocab(datasets,min_frq=3):

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
            self.data.append((torch.tensor(indices,dtype=torch.long())))
        
    def __len__(self,data):
        return len(self.data)
    
    def __getitem__(self, index):
        return super(index)


#padding
def collate_fn(batch):
    texts,label=zip(*batch)
    padded=torch.nn.utils.rnn.pad_packed_sequence(batch_first=True,padding_value=0)
    lengths=torch.tensor(len(t) for t in texts)
    label=torch.tensor(label,dtype=torch.float())
    return padded,lengths,label



#LStm classifier
class Lstmclassifier(torch.nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim,n_layers,pad_idx):

        self.embeded=torch.nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.lstm=torch.nn.LSTM(
            input_size=embed_dim,
            num_layers=n_layers,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=0.4
        )

        self.fc=torch.nn.Linear(hidden_dim*2,2)
        self.dropout=torch.nn.Dropout(0.4)

    def forward(self,text,lengths):

        embeded=self.dropout(self.embeded(text))
        packed=torch.nn.utils.rnn.pack_padded_sequence(text,batch_first=True,enforce_sorted=True)

        _,(hidden,cell)=self.lstm(packed)

        hidden=torch.cat(hidden[-2],hidden[-1],dim=1)
        hidden=self.dropout(hidden)
        return self.fc(hidden).squezze(1)
    


