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
    

#setting
vocab=build_vocab()
BATCH_SIZE=32
VOCAB_SIZE=len(vocab)
EMBEDED_DIM=100
HIDDEN_DIM=256
N_LAYERS=2
PAD_IDX=0
EPOCHS=10

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataset=IMDBdataset('train')
test_dataset=IMDBdataset('test')

train_loader=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=BATCH_SIZE,collate_fn=collate_fn)
test_loader=torch.utils.data.DataLoader(test_dataset,shuffle=False,batch_size=BATCH_SIZE,collate_fn=collate_fn)

model=Lstmclassifier(embed_dim=EMBEDED_DIM,hidden_dim=HIDDEN_DIM,n_layers=N_LAYERS,pad_idx=PAD_IDX,vocab_size=VOCAB_SIZE).to(DEVICE)

Criterion=torch.nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

#train and evaluate
def train_epoch(model,loader):
    model.train()
    total_loss,correct=0,0
    for text,label,lengths in loader:
        text,label=text.to(DEVICE),label.to(DEVICE)

        optimizer.zero_grad()
        predictions=model(text,label)
        loss=Criterion(predictions,label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
        optimizer.step()

        total_loss+=loss.item()
        pred=(torch.sigmoid(predictions)>0.5).float()
        correct+=(label==pred).sum().item()
        
        return total_loss/len(loader),correct/len(loader)
    
def evaluate(model,loader):
    model.eval()
    correct,total_loss=0,0
    with torch.no_grad():
        for text,label in loader:
            text,label=text.to(DEVICE),label.to(DEVICE)

            optimizer.zero_grad()
            predictions=model(text)
            loss=Criterion(predictions,label)

            total_loss+=loss.item()
            pred=(torch.sigmoid(predictions)>0.5).float()
            correct+=(label==pred).sum().item()

            return total_loss/len(loader) ,correct/len(loader)
        
#run 
for epoch in range(EPOCHS):
    train_loss,train_acc=train_epoch(model,train_loader)
    test_loss,test_acc=evaluate(model,test_loader)
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f"Train loss: {train_loss}|Train Acc: {train_acc}")
    print(f'Test loss: {test_loss}| Test acc: {test_acc}')

def predict(text):
    model.eval()
    tokens=tokenizer(text)
    indices=[len(t) for t in text]
    lengths=torch.tensor(len(indices))
    tensor=torch.tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits=model(tensor,lengths)
        prob=torch.sigmoid(logits).items()

    label="POS" if prob>0.5 else 'NEG'
    print(f"{text} {label} (confidence: {prob:.2f})")

predict("one scene is good but the movie is worst")
predict("this was an absolutely brilliant masterpiece")
predict("terrible boring waste of my time")
