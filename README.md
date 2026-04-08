# Sentiment Analysis with Bidirectional LSTM

Binary text classification on IMDB movie reviews — positive or negative.

## Results
| Model | Test Accuracy |
|-------|--------------|
| BiLSTM  | ~87% |

## Model Architecture
```
Input Text
↓
Tokenization 
↓
Embedding Layer 
↓
Bidirectional LSTM
↓
Concatenate
↓
Dropout (0.4)
↓
Linear Layer
↓
Sigmoid → probability (>0.5 = Positive)
```


## Concepts Covered
- Word Embeddings
- Padding & Packing sequences
- Vanishing Gradients
- LSTM Gates (forget, input, output)
- Bidirectional LSTM
- Gradient Clipping

## Setup
```bash
pip install torch datasets
python sentiment_An.py
```

## Sample Output

Epoch 1/10
Train loss: 0.66|Train Acc: 0.59
Test loss: 0.77| Test acc: 0.51

one scene is good but the movie is worst NEG (confidence: 0.01)
this was an absolutely brilliant masterpiece POS (confidence: 0.97)
terrible boring waste of my time NEG (confidence: 0.01)


## Author
**Vikas Reddy**
- GitHub: [@vikasreddy11](https://github.com/vikasreddy11)
- LinkedIn: [Vikas Reddy](https://www.linkedin.com/in/vikas-reddy-veeramreddy-26057138a)
