import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

class SequentialDataset:
    def __init__(self, csv_fn, min_length=3):
        print('Creating dataset...')
        self.min_length = min_length
        self.transactions = pd.read_csv(csv_fn)
        self.idx2article = ['<pad>'] + sorted(self.transactions['article_id'].unique().tolist())
        self.article2idx = {a: i for i, a in enumerate(self.idx2article)}
        sessions = []
        cur_user = None
        last_article = None
        cur_session = []
        for user, article in tqdm(self.transactions[['customer_id', 'article_id']].values):
            if user != cur_user:
                if len(cur_session) > 0:
                    sessions.append((cur_session, cur_user))
                cur_user = user
                cur_session = []
                last_article = None
            if article != last_article:
                cur_session.append(self.article2idx[article])
                last_article = article
        self.sessions = [s for s in sessions if len(s[0]) >= self.min_length]

    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        return self.sessions[idx]
    
def collate_fn(batch):
    x = [torch.LongTensor(item[0][:-1]) for item in batch]
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = [torch.LongTensor(item[0][1:]) for item in batch]
    y = pad_sequence(y, batch_first=True, padding_value=0)
    lengths = torch.LongTensor([len(item[0])-1 for item in batch])
    users = [item[1] for item in batch]
    return x, y, lengths, users