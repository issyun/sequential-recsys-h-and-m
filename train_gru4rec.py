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

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

    def load_checkpoint(self, fn):
        self.model.load_state_dict(torch.load(fn))
    
    def get_nll(self, x, y):
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1)
        x = x[y != 0]
        y = y[y != 0]
        nll = -torch.log(x[torch.arange(len(y)), y] + 1e-8).mean()
        num_correct = (x.argmax(dim=-1) == y).sum().item()
        return nll, num_correct
    
    def train(self, hyperparams, log_to_wandb=False, run_name=None):
        if log_to_wandb:
            wandb.init(project='recsys_term_proj', name=run_name, config=hyperparams)

        for epoch in tqdm(range(hyperparams['num_epochs'])):
            self.model.train()
            train_correct = 0
            train_total = 0
            for iter, batch in enumerate(tqdm(self.train_loader)):
                x, y, lengths, users = batch
                x = x.to(self.device)
                y = y.to(self.device)
                users = users.to(self.device)

                out = self.model(x, lengths, users)
                out = out.softmax(dim=-1)
                loss, num_correct = self.get_nll(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if log_to_wandb:
                    wandb.log({'train_loss': loss.item()})
                train_correct += num_correct
                train_total += len(y)
                if log_to_wandb and iter % 100 == 0:
                    wandb.log({'train_acc': train_correct / train_total})
                    train_correct = 0
                    train_total = 0
                if iter % 20000 == 0:
                    torch.save(self.model.state_dict(), f'checkpoints/{run_name}_epoch_{epoch}_iter_{iter}.pt')

            self.model.eval()
            val_correct = 0
            val_loss = 0
            val_total = 0
            with torch.inference_mode():
                for iter, batch in enumerate(tqdm(self.val_loader)):
                    x, y, lengths, users = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    users = users.to(self.device)

                    out = self.model(x, lengths, users)
                    out = out.softmax(dim=-1)
                    loss, num_correct = self.get_nll(out, y)
                    val_loss += loss.item()

                    if log_to_wandb:
                        wandb.log({'val_loss': loss.item()})
                    val_correct += num_correct
                    val_total += len(y)
            wandb.log({'val_acc': val_correct / val_total})
            torch.save(self.model.state_dict(), f'checkpoints/{run_name}_epoch_{epoch}_end.pt')

def main():
    dataset = SequentialDataset('data/transactions_train.csv', 'data/customers.csv')
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.95, 0.05], generator)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparams['batch_size'], shuffle=True, generator=generator, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=hyperparams['batch_size'], shuffle=False, collate_fn=collate_fn)
    model = GRU4Rec(len(dataset.idx2article), len(dataset.idx2user), hyperparams['item_emb_dim'], hyperparams['user_emb_dim'], hyperparams['hidden_dim'], hyperparams['num_layers'], hyperparams['dropout'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
    trainer = Trainer(model, train_loader, val_loader, optimizer, DEV)
    model.to(DEV)
    trainer.train(hyperparams, log_to_wandb=True, run_name='GRU4Rec_with_user_emb')

if __name__ == '__main__':
    main()