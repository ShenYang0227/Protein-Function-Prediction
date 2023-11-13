import wget
import torch
from utils import read_file,load_predicted_PDB,edge,get_embedding,protein_graph
import os

def get_PDB_structure(idx_list):  # download pdb files and obtain 3d coordinates of target proteins
    no_pdb = []
    for name in idx_list:
        name = name.split("-")[0]
        name = name.lower()
        url = "https://files.rcsb.org/download/%s.pdb" % (name)
        path = "pdb_files"

        try:
            wget.download(url, path)
            p = './pdb_files/%s.pdb' % (name)
            with open(p, 'r') as r:
                lines = r.readlines()
                with open(p, 'w') as w:
                    for line in lines:
                        if line.startswith("ATOM"):
                            w.write(line)

        except:
            print("%s not found" % name)
            no_pdb.append(name)
            continue

        finally:
            # print("finall...")
            print("END")

    return no_pdb
# train_list=read_file("./data/train.txt")
# get_PDB_structure(train_list,"valid")

def get_dataset(root):
    train_list = read_file("./data/train.txt")
    i = 1
    for name in train_list:
        name = name.split("-")[0].lower()
        file_name = "./pdb_files_train/%s.pdb" % name
        _,seq = load_predicted_PDB(file_name)
        edge_index = edge(file_name, 8.5)
        esm_embed = get_embedding(seq)
        data = protein_graph(esm_embed, edge_index)

        torch.save(data, os.path.join(root, "data_%s.pt" % i))
        i += 1

# get_dataset("./dataset/train")

