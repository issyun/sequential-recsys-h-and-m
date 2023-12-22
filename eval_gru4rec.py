import argparse
import torch
from tqdm.auto import tqdm
from models import GRU4Rec
from dataset import SequentialDataset, collate_fn

hyperparams = {
    'item_emb_dim': 128,
    'user_emb_dim': 64,
    'hidden_dim': 512,
    'num_layers': 4,
    'dropout': 0.2
}
RANDOM_SEED = 20171237
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    dataset = SequentialDataset('data/transactions_train.csv', 'data/customers.csv')
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.95, 0.05], generator)
    del(train_set)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = GRU4Rec(len(dataset.idx2item), len(dataset.idx2user), hyperparams['item_emb_dim'], hyperparams['user_emb_dim'], hyperparams['hidden_dim'], hyperparams['num_layers'], hyperparams['dropout'])
    model.to(DEV)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    num_total = args.num_tests
    reciprocal_sum = 0
    ap_sum = 0
    precision_sum = 0
    num_tests = 0
    print('Evaluating...')
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, total=num_total):
            num_tests += 1
            x, _, lengths, users = batch
            x = x.to(DEV)
            users = users.to(DEV)
            user = users[0].item()
            
            out = model(x, lengths, users)
            out = out[:, -1, :].softmax(dim=-1).squeeze()
            ranked = torch.argsort(out, descending=True).tolist()

            for rank, item in enumerate(ranked):
                if dataset.check_purchase(user, item):
                    reciprocal_sum += 1 / (rank + 1)
                    break
            
            ap = 0
            numerator = 0
            for rank, item in enumerate(ranked[:10]):
                if dataset.check_purchase(user, item):
                    numerator += 1
                    ap += numerator / (rank + 1)
            ap /= 10
            ap_sum += ap
            precision_sum += numerator / 10

            if num_tests > num_total:
                break

    print(f'MRR: {reciprocal_sum / num_tests}')
    print(f'MAP@10: {ap_sum / num_tests}')
    print(f'Precision@10: {precision_sum / num_tests}')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--num_tests', type=int, default=10000)
    argparser.add_argument('-c', '--checkpoint', type=str, default='checkpoints/GRU4Rec_ckpt.pt')
    args = argparser.parse_args()
    main(args)