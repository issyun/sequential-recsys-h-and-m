
# Sequential Recommender Models Implementation

GRU4Rec and Neural Attentive Recommendation Machine(NARM) implementation in PyTorch, with data from the Kaggle H&M Personalized Fashion Recommendations(https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/). Done as a term project for the course "Basic Recommender Systems", fall 2023.

Only the csv files from the H&M dataset is used. They should be placed in the data/ folder as so:

```
data
- articles.csv
- customers.csv
- transactions_train.csv
```

## Hyperparameters Used

### GRU4Rec

```
{
'lr': 1e-3,
'batch_size': 48,
'item_emb_dim': 128,
'user_emb_dim': 64,
'hidden_dim': 512,
'num_layers': 4,
'dropout': 0.2
}
```

### NARM

```
{
'lr': 1e-3,
'batch_size': 64,
'item_emb_dim': 128,
'hidden_dim': 512,
'num_layers': 1
}
```

## Test Results

|Model | MRR | MAP@10 | Precision@10|
|--|--|--|--|
|NARM | 0.2565 | 0.0363 | 0.0764|
|GRU4Rec(w/o user emb) | 0.4118 | 0.0733 | 0.1402|
|GRU4Rec(w/ user emb) | 0.3633 | 0.0605 | 0.1178|

These are results from preliminary tests; I belive more tests with proper hyperparameter tuning is needed.