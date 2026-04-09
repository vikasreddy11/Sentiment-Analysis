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
class RNNClassifier(torch.nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim,n_layers,pad_idx=0):
        super().__init__()

        self.embed=torch.nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.rnn=torch.nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            nonlinearity='tanh',
            dropout=0.4
        )

        self.fc=torch.nn.Linear(hidden_dim*2,1)
        self.dropout=torch.nn.Dropout(0.4)

    def forward(self,text,lengths):

        embeded=self.dropout(self.embed(text))
        packed=torch.nn.utils.rnn.pack_padded_sequence(embeded,lengths,enforce_sorted=True,batch_first=True)

        _,hidden=self.rnn(packed)

        hidden=torch.cat([hidden[-2],hidden[-1]],dim=1)
        hidden=self.dropout(hidden)

        return self.fc(hidden).squeeze(1)
    
