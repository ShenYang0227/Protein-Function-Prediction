import numpy as np
from utils import get_seq,read_file,get_edge,get_embedding,protein_graph,load_GO_annot
import os
import torch


from torch.utils.data import Dataset
from torch_geometric.data import Batch

import warnings
warnings.filterwarnings("ignore")



class GODataSet(Dataset):
    def __init__(self, root,set_type,task):
        self.root=root
        self.set_type=set_type
        self.task=task
        self.list=read_file("./data/%s.txt" %set_type)
        self.graphs=[]
        for name in self.list:
            try:
                self.graph=torch.load(os.path.join(root,"data_"+name+".pt"))
                self.graphs.append(self.graph)
            except:
                continue
                
        
        
        prot2annot, goterms, gonames, counts = load_GO_annot("./data/nrPDB-GO_2019.06.18_annot.tsv")
        self.y_trues = np.stack([prot2annot[pdb_c][self.task] for pdb_c in self.list])
        self.y_trues = torch.tensor(self.y_trues)
        
        
    def __getitem__(self, idx):
        return self.graphs[idx],self.y_trues[idx]

    def __len__(self):
        return len(self.graphs)

#train_set=GODataSet("./dataset/train","train","cc")