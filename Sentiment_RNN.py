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
    
#setting 
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE=32
EMBED_DIM=100
HIDDEN_DIM=256
N_LAYERS=2
EPOCHS=10

from datasets import load_dataset
raw           = load_dataset('imdb')
vocab         = build_vocab(raw['train'])
train_dataset = IMDBdataset(raw['train'], vocab)
test_dataset  = IMDBdataset(raw['test'],  vocab)
VOCAB_SIZE = len(vocab)     

train_loader=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=BATCH_SIZE,collate_fn=collate_fn)
test_loader=torch.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=BATCH_SIZE,collate_fn=collate_fn)


model=RNNClassifier(embed_dim=EMBED_DIM,hidden_dim=HIDDEN_DIM,n_layers=N_LAYERS,vocab_size=VOCAB_SIZE).to(DEVICE)

Criterion=torch.nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

#train
def train_epoch(model,loader):
    model.train()
    correct,total_loss=0,0
    for text,lengths,label in loader:
        text,label=text.to(DEVICE),label.to(DEVICE)

        optimizer.zero_grad()
        predictions=model(text,lengths)
        loss=Criterion(predictions,label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
        optimizer.step()

        total_loss+=loss.item()
        pred=(torch.sigmoid(predictions)>0.5).float
        correct+=(pred==label).sum().item()
    
    return total_loss/len(loader),correct/len(loader)

#evaluate
def evaluate(model,loader):
    model.eval()
    correct,total_loss=0,0
    with torch.no_grad():   
        for text,lengths,label in loader:
            text,label =text.to(DEVICE),label.to(DEVICE)

            optimizer.zero_grad()
            predictions=model(text,lengths)
            loss=Criterion(predictions,label)

            total_loss+=loss.item()
            pred=(torch.sigmoid(predictions)>0.5).float
            correct+=(pred==label).sum().item()

    return total_loss/len(loader), correct/len(loader)

#Run
for epoch in range(EPOCHS):
    train_loss,train_acc=train_epoch(model,train_loader)
    test_loss,test_acc=evaluate(model,test_loader)
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'train_loss: {train_loss:.2f} | train_acc: {train_acc:.2f}')
    print(f'test_loss: {test_loss:.2f} | test_acc: {test_acc:.2f}')

def predict(text):
    model.eval()
    tokens=tokenizer(text)
    indices=[vocab.get(t,'<unk>') for t in tokens]
    lengths=torch.tensor(len(indices))
    tensor=torch.tensor(lengths)

    with torch.no_grad():
        logits=model(text,lengths)
        prob=torch.sigmoid(logits).item()

    label="Positive" if prob > 0.5 else 'Negitive'
    print(f"{text} {label} (confidence: {prob:.2f})")