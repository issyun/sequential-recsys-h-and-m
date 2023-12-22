import argparse
import torch
from torch.nn.utils.rnn import unpad_sequence
from tqdm.auto import tqdm
import wandb
from models import NARM
from dataset import SequentialDataset, collate_fn

hyperparams = {
    'num_epochs': 4,
    'lr': 0.001,
    'batch_size': 64,
    'item_emb_dim': 128,
    'hidden_dim': 512,
    'num_layers': 1
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
        '''
        x: (batch_size, num_items)
        y: (batch_size)
        '''
        nll = -torch.log(x[torch.arange(x.shape[0]), y] + 1e-8).mean()
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
                x, y, lengths, _ = batch
                x = x.to(self.device)
                y = torch.LongTensor([s[-1] for s in unpad_sequence(y, lengths, batch_first=True)]).to(self.device)

                out = self.model(x, lengths).softmax(dim=-1)
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
                    x, y, lengths, _ = batch
                    x = x.to(self.device)
                    y = torch.LongTensor([s[-1] for s in unpad_sequence(y, lengths, batch_first=True)]).to(self.device)

                    out = self.model(x, lengths).softmax(dim=-1)
                    loss, num_correct = self.get_nll(out, y)
                    val_loss += loss.item()
                    val_correct += num_correct
                    val_total += len(y)

            if log_to_wandb:
                wandb.log({'val_loss': val_loss / val_total})
                wandb.log({'val_acc': val_correct / val_total})

            torch.save(self.model.state_dict(), f'checkpoints/{run_name}_epoch_{epoch}_end.pt')

def main(args):
    dataset = SequentialDataset('data/transactions_train.csv', 'data/customers.csv')
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.95, 0.05], generator)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparams['batch_size'], shuffle=True, generator=generator, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=hyperparams['batch_size'], shuffle=False, collate_fn=collate_fn)
    model = NARM(len(dataset.idx2item), hyperparams['item_emb_dim'], hyperparams['hidden_dim'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
    trainer = Trainer(model, train_loader, val_loader, optimizer, DEV)
    model.to(DEV)
    trainer.train(hyperparams, log_to_wandb=True, run_name=args.run_name)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--run_name', type=str, default='NARM')
    args = argparser.parse_args()
    main(args)