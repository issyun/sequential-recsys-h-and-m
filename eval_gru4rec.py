import torch
from tqdm.auto import tqdm
import wandb
from models import GRU4Rec
from dataset import SequentialDataset, collate_fn

hyperparams = {
    'num_epochs': 5,
    'lr': 0.001,
    'batch_size': 48,
    'item_emb_dim': 128,
    'user_emb_dim': 64,
    'hidden_dim': 512,
    'num_layers': 4,
    'dropout': 0.2
}
RANDOM_SEED = 20171237
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    dataset = SequentialDataset('data/transactions_train.csv', 'data/customers.csv')
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.95, 0.05], generator)
    del(train_set)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=hyperparams['batch_size'], shuffle=False, collate_fn=collate_fn)
    model = GRU4Rec(len(dataset.idx2article), len(dataset.idx2user), hyperparams['item_emb_dim'], hyperparams['user_emb_dim'], hyperparams['hidden_dim'], hyperparams['num_layers'], hyperparams['dropout'])
    model.to(DEV)
    model.load_state_dict(torch.load('checkpoints/GRU4Rec_with_user_emb_epoch_2_end.pt'))