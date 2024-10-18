from NEWDGLDataSet import GoTermDataset, collate_fn
# from torch.utils.data import DataLoader
from utils import log
from sklearn import metrics
# from config import get_config
from megraph.utils import my_tqdm
from Bio.PDB.PDBParser import PDBParser
# from Loss.MultiGrainedFocalLoss import *

import os
import sys
import torch
import csv
import pickle
import esm
import argparse

from torch_geometric import *
import dgl
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import to_networkx
import networkx as nx

sys.path.append(os.getcwd())

import argparse
import traceback

import numpy as np
import taichi as ti
from megraph.args_utils import get_args_and_model
from megraph.layers import layer_factory, register_layers_args
from megraph.logger import logger
from megraph.models import model_factory, register_models_args
from megraph.torch_utils import get_num_params, set_global_seed
from megraph.trainer import Trainer
from megraph.utils import register_logging_args, set_logger
from dgl.dataloading import GraphDataLoader, DataLoader, ClusterGCNSampler

parser = argparse.ArgumentParser(description="Predict Protein Function")
register_logging_args(parser)
register_layers_args(parser)
register_models_args(parser)

parser.add_argument("--epochs", "-ep", type=int, default=200, help="number of epochs")
parser.add_argument("--dataset", "-ds", type=str, default='test', help='test or AF2test')
parser.add_argument("--train_num", "-tn", type=int, default=16, help='number of train model')
parser.add_argument("--sub_function", "-sf", type=str, default="bp", help="class of function")
parser.add_argument('--pdb', type=str, default='./data/4RQ2-A.pdb', help='Input the query pdb file')
parser.add_argument('--prob', default=0.5, type=float, help='Output the function with predicted probility > 0.5 .')

parser.add_argument("--seed", "-se", type=int, default=2024)
parser.add_argument(
    "--configs-dir", "-cd", type=str, default="configs", help="configs dir"
)
parser.add_argument(
    "--config-file", "-cfg", type=str, default="./configs/best/cfg_protein_bp.py", help="config filename"
)
parser.add_argument("--runs", "-rn", type=int, default=10, help="number of runs")
parser.add_argument("--record-graphs", "-rg", action="store_true", help="record graphs")
parser.add_argument("--save-model", "-sm", action="store_true", help="save model")
parser.add_argument(
    "--load-model-path", "-lmp", type=str, default=None, help="load model"
)
parser.add_argument("--debug", "-de", action="store_true", help="debug")
# parser.add_argument("--debug", "-de", action="store_true", help="debug")
parser.add_argument("--gpu-id", "-gid", type=int, default=0, help="gpu id")
parser.add_argument("--dataset_name", "-dname", type=str, default="protein", help="The input dataset")
parser.add_argument("--dataset_subname", "--dsub", type=str, default=None, help="The name for the sub dataset, if applicable")

args, graph_layer, graph_model = get_args_and_model(
    parser, layer_factory, model_factory
)

set_global_seed(args)
dump_dir = set_logger(args)
save_dir = "/mnt/c/2024.5.20_dme"
record_graphs_dir = None
if args.record_graphs:
    if args.model == "megraph":
        record_graphs_dir = os.path.join(save_dir, "graphs")
    else:
        args.record_graphs = False
        logger.info("Only megraph model need record graphs")

save_model_dir = None
if args.save_model:
    save_model_dir = os.path.join(save_dir, "models")
    os.mkdir(save_model_dir)

ti.init(random_seed=args.seed)

def load_GO_annot(filename):
    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        next(reader, None)
        goterms[
            onts[0]] = next(reader)
        next(reader, None)
        gonames[onts[0]] = next(reader)

        next(reader, None)
        goterms[onts[1]] = next(reader)
        next(reader, None)
        gonames[onts[1]] = next(reader)

        next(reader, None)
        goterms[onts[2]] = next(reader)
        next(reader, None)
        gonames[onts[2]] = next(reader)

        next(reader, None)
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

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False

_, goterms, gonames, _ = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")

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

def aa2idx(seq):
    # convert letters into numbers
    abc = np.array(list("ARNDCQEGHILKMFPSTWYVX"), dtype='|S1').view(np.uint8)
    idx = np.array(list(seq), dtype='|S1').view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > 20] = 20
    return idx

restype_3to1 = {v: k for k, v in restype_1to3.items()}

pdb = args.pdb
parser = PDBParser()

struct = parser.get_structure("x", pdb)
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
        sequence += restype_3to1[chain[(' ', idx, ' ')].get_resname()]  # 尝试获取残基的名称，并添加到sequence中。
    except:
        sequence += 'X'

print(f'sequence={sequence}')
sequence = sequence[:1000]

Ca_array = np.array(Ca_array)

resi_num = Ca_array.shape[0]
G = np.dot(Ca_array, Ca_array.T)
H = np.tile(np.diag(G), (resi_num, 1))
dismap = (H + H.T - 2 * G) ** 0.5
len_0 = dismap.shape[0]
if len_0 > 1000:
    dismap = dismap[:1000, :1000]

device = args.device

esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(device)

esm_model.eval()

batch_labels, batch_strs, batch_tokens = batch_converter([('tmp', sequence)])
batch_tokens = batch_tokens.to(device)
with torch.no_grad():
    results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33][0].cpu().numpy().astype(np.float16)
    esm_embed = token_representations[1:len(sequence) + 1]


row, col = np.where(dismap <= 10)
edge = [row, col]

seq_code = aa2idx(sequence)
seq_code = torch.IntTensor(seq_code)
edge_index = torch.LongTensor(edge)
data = Data(x=torch.from_numpy(esm_embed), edge_index=edge_index, native_x=seq_code).to_dict()
data = Data.from_dict(data)
G = to_networkx(data)
edge_attrs = []
G = dgl.from_networkx(G, edge_attrs=edge_attrs)
G = dgl.add_self_loop(G)
G.ndata["feat"] = data.x
G.ndata["native_x"] = data.native_x
batch = dgl.batch([G])

task = 'gpred'
input_dims = [0, 512, 0]
output_dims_All = [1943, 0, 0]
output_dims_Head = output_dims_All.copy()
output_dims_Tail = output_dims_All.copy()
if args.sub_function == 'bp':
    output_dims_All[0] = 1943
    output_dims_Head[0] = 300
    output_dims_Tail[0] = 1643
elif args.sub_function == 'mf':
    output_dims_All[0] = 489
    output_dims_Head[0] = 100
    output_dims_Tail[0] = 389
elif args.sub_function == 'cc':
    output_dims_All[0] = 320
    output_dims_Head[0] = 50
    output_dims_Tail[0] = 270
pe_dim = 0

if args.layer in ["gcn"]:
    input_dims[2] = 0

if args.model in ["plain", "megraph", "unet", "hgnet"]:

    def build_conv(**kwargs):
        return graph_layer.from_args(args, **kwargs)


    model_All = graph_model.from_args(
        args,
        input_dims=input_dims,
        output_dims=output_dims_All,
        pe_dim=pe_dim,
        task=task,
        embed_method={},
        build_conv=build_conv,
    )
model_All = model_All.to(args.device)
checkpoint = torch.load(f'./Model/best_{args.sub_function}_All.pt',
                        map_location=args.device)
model_All.load_state_dict(checkpoint['model_state_dict'])
model_All.eval()
with torch.no_grad():
    model_All.perturbed = False
    model_All.set_perturbed(False)
    out1, out2 = model_All(batch.to(args.device))
    out_All = 0.5 * out1 + 0.5 * out2
    out_All = torch.sigmoid(out_All)


if args.model in ["plain", "megraph", "unet", "hgnet"]:

    def build_conv(**kwargs):
        return graph_layer.from_args(args, **kwargs)


    model_Head = graph_model.from_args(
        args,
        input_dims=input_dims,
        output_dims=output_dims_Head,
        pe_dim=pe_dim,
        task=task,
        embed_method={},
        build_conv=build_conv,
    )
model_Head = model_Head.to(args.device)
checkpoint = torch.load(f'./Model/best_{args.sub_function}_Head.pt',
                        map_location=args.device)
model_Head.load_state_dict(checkpoint['model_state_dict'])
model_Head.eval()
with torch.no_grad():
    model_Head.perturbed = False
    model_Head.set_perturbed(False)
    out1, out2 = model_Head(batch.to(args.device))
    out_Head = 0.5 * out1 + 0.5 * out2
    out_Head = torch.sigmoid(out_Head)

if args.model in ["plain", "megraph", "unet", "hgnet"]:

    def build_conv(**kwargs):
        return graph_layer.from_args(args, **kwargs)


    model_Tail = graph_model.from_args(
        args,
        input_dims=input_dims,
        output_dims=output_dims_Tail,
        pe_dim=pe_dim,
        task=task,
        embed_method={},
        build_conv=build_conv,
    )
model_Tail = model_Tail.to(args.device)
checkpoint = torch.load(f'./Model/best_{args.sub_function}_Tail.pt',
                        map_location=args.device)
model_Tail.load_state_dict(checkpoint['model_state_dict'])
model_Tail.eval()
with torch.no_grad():
    model_Tail.perturbed = False
    model_Tail.set_perturbed(False)
    out1, out2 = model_Tail(batch.to(args.device))
    out_Tail = 0.5 * out1 + 0.5 * out2
    out_Tail = torch.sigmoid(out_Tail)


distribution_path = f"./data/DMETrain/distribution_{args.sub_function}_All.txt"
with open(distribution_path) as f:
    for line in f:
        list_temp = line.replace(" ", "").split(",")

list_distribution = list(map(int, list_temp))
label_counts_np = np.array(list_distribution)
sorted_indices = np.argsort(-label_counts_np)

if args.sub_function == "bp":
    indices_after_50 = sorted_indices[0:300]
elif args.sub_function == "cc":
    indices_after_50 = sorted_indices[0:50]
else:
    indices_after_50 = sorted_indices[0:100]
sorted_indices_after_50 = np.sort(indices_after_50)
length = len(sorted_indices_after_50)
print(f'length={length}')
for i in range(0, length):
    index = sorted_indices_after_50[i]
    out_All[:, index] = (out_All[:, index] + out_Head[:, i])/2.0
    out_All[:, index] = (out_All[:, index] + out_Head[:, i])/2.0

if args.sub_function == "bp":
    indices_after_50_t = sorted_indices[300:]
elif args.sub_function == "cc":
    indices_after_50_t = sorted_indices[50:]
else:
    indices_after_50_t = sorted_indices[100:]
sorted_indices_after_50_t = np.sort(indices_after_50_t)
length = len(sorted_indices_after_50_t)
print(f'length={length}')
for i in range(0, length):
    index = sorted_indices_after_50_t[i]
    out_All[:, index] = (out_All[:, index] + out_Tail[:, i])/2.0
    out_All[:, index] = (out_All[:, index] + out_Tail[:, i])/2.0

y_pred = out_All
func_index = np.where(y_pred.cpu().numpy() > args.prob)[1]
if func_index.shape[0] == 0:
    print(f'Sorry, no functions of {args.sub_function.upper()} can be predicted...')
else:
    print(f'The protein may hold the following functions of {args.sub_function.upper()}:')
    for idx in func_index:
        print(
            f'Possibility: {round(float(y_pred[0][idx]), 2)} ||| Functions: {goterms[args.sub_function][idx]}, {gonames[args.sub_function][idx]}')

# python Predictor.py  --sub_function cc --config-file ./configs/best/cfg_protein_cc.py --pdb ./data/4RQ2-A.pdb --prob 0.5
