import torch
from torch_geometric.data import DataLoader
from dataset import GODataSet

import torch.nn as nn

from sklearn import metrics
import matplotlib.pyplot as plt

import copy
import argparse

from model import esm_model


def train(out_dims,gc_dims,gc_layers,task,num_epochs,lr):
    # 1, 加载数据集
    train_set = GODataSet("./dataset/train", "train", task)
    pos_weights=train_set.pos_weights.to("cuda")
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    valid_set = GODataSet("./dataset/valid", "valid", task)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True, num_workers=2)

    device = torch.device("cuda")
    model = esm_model(out_dims=out_dims,gc_dims=gc_dims,gc_layers=gc_layers).to(device)

    loss_fn=nn.BCELoss(weight=pos_weights)
    optim=torch.optim.Adam(model.parameters(),lr=lr)

    for epoch in range(num_epochs):
        print("第{}次训练".format(epoch+1))
        train_loss_all=[]
        for batch in train_loader:
            data=batch[0].to(device)
            label=batch[1].to(device)
            label = torch.tensor(label, dtype=torch.float).to(device)

            model.train()
            out=model(data)

            loss=loss_fn(out,label)
            loss=loss.mean()
            train_loss_all.append(loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

        print("train loss:{:.4}".format(train_loss_all[-1]))

        valid_loss_all=[]
        y_true_all=[]
        y_pred_all=[]
        valid_acc_all=[]
        best_acc=0.0

        with torch.no_grad():
            for batch in valid_loader:
                data=batch[0].to(device)
                label=batch[1].to(device)
                label = torch.tensor(label, dtype=torch.float)
                y_true_all.append(label)

                model.eval()
                out=model(data)


                y_pred=torch.where(out >= 0.5, 1, 0)
                y_pred_all.append(y_pred)
                valid_loss=loss_fn(out,label).mean()
                valid_loss_all.append(valid_loss)

            y_pred_all = torch.cat(y_pred_all, dim=0).cpu()
            y_true_all = torch.cat(y_true_all, dim=0).cpu()

            print("valid loss:{:.4}".format(valid_loss_all[-1]))


            acc=metrics.average_precision_score(y_true_all.cpu().numpy(),y_pred_all.cpu().numpy(),average="samples")
            print("acc",acc)
            valid_acc_all.append(acc)


        if valid_acc_all[-1]>best_acc:
            best_acc=valid_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())

    torch.save(best_model_wts,f"{gc_layers+task}.pt")



def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,"ro-",label="train loss")
    plt.plot(train_process["epoch"],train_process.valid_loss_all,"bs-",label="valid loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)

    plt.plot(train_process["epoch"], train_process.valid_acc_all, "bs-", label="valid accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.show()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("__task",type=str,default="cc",choices=["bp","mf","cc"])
    p.add_argument("--out_dims",type=int,default=320)
    p.add_argument("--gc_dims",type=int,default=[2560,256,128,512])
    p.add_argument("--gc_layers",type=str,default="GCN")
    p.add_argument("--num_epochs",type=int,default=50)
    p.add_argument("--lr",type=float,default=1e-3)

    args=p.parse_args()
    train(args.out_dims,args.gc_dims,args.gc_layers,args.num_epochs,args.lr)













