import torch
from torch_geometric.data import DataLoader
from dataset import GODataSet
from train import esm_model

from sklearn import metrics
import torch.nn as nn
import argparse

def test(out_dims,gc_dims,gc_layers,task):
    test_set = GODataSet("./dataset/test", "test", task)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)

    device = torch.device("cuda")
    model = esm_model(out_dims=out_dims, gc_dims=gc_dims, gc_layers=gc_layers).to(device)
    model.load_state_dict(torch.load(f"{gc_layers+task}.pt"))
    model.eval()

    eval_loss_all=[]
    y_pred_all=[]
    y_true_all=[]
    loss_fn=nn.BCELoss()
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0].to(device)
            label = batch[1].to(device)
            y_true_all.append(label)

            model.eval()
            out = model(data)

            y_pred_all.append(out)
            eval_loss = loss_fn(out, label)
            eval_loss=eval_loss.mean()
            print("eval_loss",eval_loss)
            eval_loss_all.append(eval_loss)

        y_pred_all=torch.cat(y_pred_all,dim=0).cpu()
        y_true_all=torch.cat(y_true_all,dim=0).cpu()

        aupr=metrics.average_precision_score(y_true_all.numpy(),y_pred_all.numpy())
        print("aupr:{:.4f}",aupr)
        f1=metrics.f1_score(y_true_all.cpu().numpy(),y_pred_all.cpu().numpy())
        print("f1:{:.4f}",f1)



if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("__task",type=str,default="cc",choices=["bp","mf","cc"])
    p.add_argument("--out_dims",type=int,default=320)
    p.add_argument("--gc_dims",type=int,default=[2560,256,128,512])
    p.add_argument("--gc_layers",type=str,default="GCN")


    args=p.parse_args()
    test(args.out_dims,args.gc_dims,args.gc_layers)

