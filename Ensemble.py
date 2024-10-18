import numpy as np
import pickle
import csv

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble predictions for different GO ontologies.")
    parser.add_argument('--ontology', type=str, choices=['bp', 'cc', 'mf'], required=True, help='Choose the GO ontology to process: bp, cc, or mf.')
    return parser.parse_args()

args = parse_args()
ontology = args.ontology


def load_GO_annot(filename):
    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        next(reader, None)
        goterms[onts[0]] = next(reader)
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

prot2annot, goterms, gonames, counts = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")

distribution_path = f"./data/DMETrain/distribution_{ontology}_All.txt"
with open(distribution_path) as f:
    for line in f:
        list_temp = line.replace(" ", "").split(",")

list_distribution = list(map(int, list_temp))
label_counts_np = np.array(list_distribution)
sorted_indices = np.argsort(-label_counts_np)  # 从大到小排序

# 选出排名第50之后的所有元素的下标ID
if ontology == "bp":
    indices_after_50 = sorted_indices[0:300]
elif ontology == "cc":
    indices_after_50 = sorted_indices[0:50]
else:
    indices_after_50 = sorted_indices[0:100]
sorted_indices_after_50 = np.sort(indices_after_50)

if ontology == "bp":
    indices_after_50_t = sorted_indices[300:]
elif ontology == "cc":
    indices_after_50_t = sorted_indices[50:]
else:
    indices_after_50_t = sorted_indices[100:]
sorted_indices_after_50_t = np.sort(indices_after_50_t)

# Ensemble_Tail

Predicton_PDB_All = f"/mnt/c/All-{ontology}/PredictionPkl/PDB_{ontology}_test.pkl"
with open(Predicton_PDB_All, "rb") as f:
    predict_pdb_All = pickle.load(f)

Predicton_AF2_All = f"/mnt/c/All-{ontology}/PredictionPkl/AF2_{ontology}_test.pkl"
with open(Predicton_AF2_All, "rb") as f:
    predict_AF2_All = pickle.load(f)


Predicton_PDB_Tail = f"/mnt/c/Tail-{ontology}/PredictionPkl/PDB_{ontology}_test.pkl"
with open(Predicton_PDB_Tail, "rb") as f:
    predict_pdb_Tail = pickle.load(f)

Predicton_AF2_Tail = f"/mnt/c/Tail-{ontology}/PredictionPkl/AF2_{ontology}_test.pkl"
with open(Predicton_AF2_Tail, "rb") as f:
    predict_AF2_Tail = pickle.load(f)

length = len(sorted_indices_after_50_t)
print(f'length={length}')
for i in range(0, length):
    index = sorted_indices_after_50_t[i]
    predict_pdb_All['Y_pred'][:, index] = (predict_pdb_All['Y_pred'][:, index] + predict_pdb_Tail['Y_pred'][:, i])/2.0
    predict_AF2_All['Y_pred'][:, index] = (predict_AF2_All['Y_pred'][:, index] + predict_AF2_Tail['Y_pred'][:, i])/2.0

    predict_pdb_path = f"/mnt/c/All-{ontology}/PredictionPkl/Union_PDB_{ontology}_test_tail.pkl"
    with open(predict_pdb_path, "wb") as f:
        pickle.dump(predict_pdb_All, f)

    predict_AF2_path = f"/mnt/c/All-{ontology}/PredictionPkl/Union_AF2_{ontology}_test_tail.pkl"
    with open(predict_AF2_path, "wb") as f:
        pickle.dump(predict_AF2_All, f)

# enemble_Head

Predicton_PDB_All = f"/mnt/c/All-{ontology}/PredictionPkl/Union_PDB_{ontology}_test_tail.pkl"
with open(Predicton_PDB_All, "rb") as f:
    predict_pdb_All = pickle.load(f)

Predicton_AF2_All = f"/mnt/c/All-{ontology}/PredictionPkl/Union_AF2_{ontology}_test_tail.pkl"
with open(Predicton_AF2_All, "rb") as f:
    predict_AF2_All = pickle.load(f)


Predicton_PDB_Head = f"/mnt/c/Head-{ontology}/PredictionPkl/PDB_{ontology}_test.pkl"
with open(Predicton_PDB_Head, "rb") as f:
    predict_pdb_Head = pickle.load(f)

Predicton_AF2_Head = f"/mnt/c/Head-{ontology}/PredictionPkl/AF2_{ontology}_test.pkl"
with open(Predicton_AF2_Head, "rb") as f:
    predict_AF2_Head = pickle.load(f)

length = len(sorted_indices_after_50)
print(f'length={length}')
for i in range(0, length):
    index = sorted_indices_after_50[i]
    predict_pdb_All['Y_pred'][:, index] = (predict_pdb_All['Y_pred'][:, index] + predict_pdb_Head['Y_pred'][:, i])/2.0
    predict_AF2_All['Y_pred'][:, index] = (predict_AF2_All['Y_pred'][:, index] + predict_AF2_Head['Y_pred'][:, i])/2.0

    predict_pdb_path = f"/mnt/c/All-{ontology}/PredictionPkl/Union_PDB_{ontology}_test_all.pkl"
    with open(predict_pdb_path, "wb") as f:
        pickle.dump(predict_pdb_All, f)

    predict_AF2_path = f"/mnt/c/All-{ontology}/PredictionPkl/Union_AF2_{ontology}_test_all.pkl"
    with open(predict_AF2_path, "wb") as f:
        pickle.dump(predict_AF2_All, f)

# python ./Ensemble.py --ontology cc