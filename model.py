import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv,SAGEConv,ChebConv
import warnings
from torch_geometric.nn import global_mean_pool as gmp


class esm_model(nn.Module):
    def __init__(self,out_dims,gc_dims,gc_layers):
        super(esm_model,self).__init__()
        if gc_layers=="GCN":
            self.GConv=GCNConv
        elif gc_layers=="GAT":
            self.GConv=GATConv
        elif gc_layers=="SAGEConv":
            self.GConv=SAGEConv
        elif gc_layers=="ChebConv":
            self.GConv=ChebConv
        else:
            warnings.warn("gc_layers not specified! No GraphConv used!")

        print("*********ESM_GNN model with %s layers*********"%(gc_layers))

        #create GNN layers
        self.relu=nn.ReLU()
        self.gnn_list=[]
        for i in range(len(gc_dims)-1):
            x=self.GConv(gc_dims[i],gc_dims[i+1])
            self.gnn_list.append(x)

        self.fc = nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,out_dims),
            nn.Dropout(0.3),
            nn.BatchNorm1d(out_dims),
            nn.Sigmoid())


    def forward(self,data):
        x,edge_index,batch=data.x,data.edge_index,data.batch

        out_list=[x]
        for l in range(len(self.gnn_list)):
            out=self.gnn_list[l].to("cuda").forward(out_list[l],edge_index)
            out=self.relu(out)
            out_list.append(out)

        out = gmp(out_list[-1], batch)
        out=self.fc(out)

        return out
