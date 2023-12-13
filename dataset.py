import pandas as pd
import torch
from tqdm.auto import tqdm

class HMDataset:
    def __init__(self, csv_fn, min_length=3):
        print('Creating dataset...')
        self.min_length = min_length
        self.transactions = pd.read_csv(csv_fn)
        self.idx2article = sorted(self.transactions['article_id'].unique().tolist())
        self.article2idx = {a: i for i, a in enumerate(self.idx2article)}
        sessions = []
        cur_user = None
        cur_session = []
        for user, article in tqdm(self.transactions[['customer_id', 'article_id']].values):
            if user != cur_user:
                if len(cur_session) > 0:
                    sessions.append(cur_session)
                cur_user = user
                cur_session = []
            cur_session.append(self.article2idx[article])
        self.sessions = [s for s in sessions if len(s) >= self.min_length]

    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sessions[idx][:-1]), torch.tensor(self.sessions[idx][1:])