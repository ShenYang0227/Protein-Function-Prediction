import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
import esm
import torch
from torch_geometric.data import Data, Batch
import csv
import wget
import os
from datetime import datetime


import warnings
warnings.filterwarnings("ignore")


def read_file(filename):
    train_list=[]
    with open (filename) as f:
        for line in f:
            train=line.split("\n")[0]
            train_list.append(train)
    return train_list

def get_seq(filename,name):
    with open(filename) as f:
        for line in f:
            line = line.split("\n")[0]
            idx = line.split("\t")[0]
            seq = line.split("\t")[-1]
            if idx==name:
                return seq

def get_edge(path,name):
    pdb_name=name.split("-")[0].lower()+".pdb"
    pdb_path = os.path.join(path, pdb_name)
    edge_index = edge(pdb_path, 8.5)
    return edge_index


def load_FASTA(filename):  #打印蛋白质名称，序列
    # Loads fasta file and returns a list of the Bio SeqIO records
    infile = open(filename, 'rU')
    id_index = []
    proteins = []
    for entry in SeqIO.parse(infile, 'fasta'):
        proteins.append(str(entry.seq))
        id_index.append(str(entry.id))
    return id_index,proteins

def get_PDB_structure(idx_list):  # download pdb and add it in files
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
            print("finall...")
            print("END")

    return no_pdb

def load_predicted_PDB(pdbfile):##生成contact maps的距离矩阵 # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    # distances = np.empty((len(residues), len(residues)))
    # for x in range(len(residues)):
    #     for y in range(len(residues)):
    #         one = residues[x]["CA"].get_coord()
    #         two = residues[y]["CA"].get_coord()
    #         distances[x, y] = np.linalg.norm(one-two)

    return seqs[0]

def edge(filename, camp_threshold):
    restype_1to3 = {
        'A': 'ALA',
        'R': 'ARG',
        'N': 'ASN',
        'D': 'ASP',
        'C': 'CYS',
        'Q': 'GLN',
        'E': 'GLU',
        'G': 'GLY',
        'H': 'HIS',
        'I': 'ILE',
        'L': 'LEU',
        'K': 'LYS',
        'M': 'MET',
        'F': 'PHE',
        'P': 'PRO',
        'S': 'SER',
        'T': 'THR',
        'W': 'TRP',
        'Y': 'TYR',
        'V': 'VAL',
    }

    restype_3to1 = {v: k for k, v in restype_1to3.items()}
    parser = PDBParser()
    struct = parser.get_structure(file=filename, id=None)
    model = struct[0]
    chain_id = list(model.child_dict.keys())[0]
    chain = model[chain_id]
    Ca_array = []
    sequence = ''
    seq_idx_list = list(chain.child_dict.keys())
    seq_len = seq_idx_list[-1][1] - seq_idx_list[0][1] + 1

    for idx in range(seq_idx_list[0][1], seq_idx_list[-1][1] + 1):
        try:
            Ca_array.append(chain[(' ', idx, ' ')]['CA'].get_coord())
        except:
            Ca_array.append([np.nan, np.nan, np.nan])
        try:
            sequence += restype_3to1[chain[(' ', idx, ' ')].get_resname()]
        except:
            sequence += 'X'

    # print(sequence)
    Ca_array = np.array(Ca_array)

    resi_num = Ca_array.shape[0]
    G = np.dot(Ca_array, Ca_array.T)
    H = np.tile(np.diag(G), (resi_num, 1))
    dismap = (H + H.T - 2 * G) ** 0.5

    row, col = np.where(dismap <= camp_threshold)
    edge = [row, col]
    return edge

def get_embedding(Sequence):
    esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([('tmp', Sequence)])

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33][0].cpu().numpy().astype(np.float16)
        esm_embed = token_representations[1:len(Sequence) + 1]
    return esm_embed

def load_GO_annot(filename):  ##加载注释文件
    # Load GO annotations
    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if goterm != '']
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0
    return prot2annot, goterms, gonames, counts

def load_EC_annot(filename):
    # Load EC annotations """
    prot2annot = {}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = {'ec': next(reader)}
        next(reader, None)  # skip the headers
        counts = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            ec_indices = [ec_numbers['ec'].index(ec_num) for ec_num in prot_ec_numbers.split(',')]
            prot2annot[prot] = {'ec': np.zeros(len(ec_numbers['ec']), dtype=np.int64)}
            prot2annot[prot]['ec'][ec_indices] = 1.0
            counts['ec'][ec_indices] += 1
    return prot2annot, ec_numbers, ec_numbers, counts

def protein_graph(esm_embed, edge_index):  #加载GCN的输入文件，用在predictor.py


    # add edge to pairs whose distances are more possible under 8.25
    #row, col = edge_index
    edge_index = torch.LongTensor(edge_index)
    # if AF_embed == None:
    #     data = Data(x=seq_code, edge_index=edge_index)
    # else:
    x=torch.tensor(esm_embed,dtype=torch.float)
    # y=torch.tensor(y,dtype=torch.float)
    # y=torch.reshape(y,(1,320))

    data = Data(x=x, edge_index=edge_index)
    return data


def log(*args):
    print(f'[{datetime.now()}]', *args)
