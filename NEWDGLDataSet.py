import torch
from dgl.data import DGLDataset
from torch_geometric.data import Data, Batch
import pickle as pkl
from utils import load_GO_annot
import numpy as np
import os
# from utils import aa2idx
import sys
# import esm
import dgl
import pickle
from dgl.data.utils import download, load_graphs, save_graphs

def collate_fn(batch):
    graphs, y_trues = map(list, zip(*batch))
    return dgl.batch(graphs), torch.stack(y_trues).float()


class GoTermDataset(DGLDataset):

    def __init__(
        self,
        set_type="test",
        task="bp",
        AF2model=False,
        Train_part="All",
        name="PPF",
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
        part_data=True,
    ):
        self.set_type = set_type
        self.task = task
        self.AF2model = AF2model
        self.Train_part = Train_part
        self.part_data = part_data
        self._url = None
        self.path = f"../MyProjectData/DME{self.task}/DME{self.set_type}"

        super().__init__(
            name=name,
            url=self._url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


    def has_cache(self):
        graph_path = os.path.join(self.path, "0.bin")
        return os.path.exists(graph_path)

    def load(self):
        if self.set_type != 'AF2test':
            prot2annot, goterms, gonames, counts = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")
        else:
            prot2annot, goterms, gonames, counts = load_GO_annot('data/nrSwiss-Model-GO_annot.tsv')
        goterms = goterms[self.task]
        gonames = gonames[self.task]
        output_dim = len(goterms)
        class_sizes = counts[self.task]
        mean_class_size = np.mean(class_sizes)
        pos_weights = mean_class_size / class_sizes
        pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights))

        self.pos_weights = torch.tensor(pos_weights).float()


        self.processed_dir = f"../MyProjectData/DME{self.task}/DME{self.set_type}"
        if self.Train_part == "All":
            self.label_processed_dir = f"../MyProjectData/DME{self.task}/DME{self.set_type}/labels_{self.task}.pkl"
        elif self.Train_part == "Head":
            self.label_processed_dir = f"../MyProjectData/DME{self.task}/DME{self.set_type}/labels_Head_{self.task}.pkl"
        else:
            self.label_processed_dir = f"../MyProjectData/DME{self.task}/DME{self.set_type}/labels_Long_Tail_{self.task}.pkl"

        self.processed_dir_AF2 = f"../MyProjectData/DMEAF2{self.task}/DME{self.set_type}"
        if self.Train_part == "All":
            self.label_processed_dir_AF2 = f"../MyProjectData/DMEAF2{self.task}/DME{self.set_type}/labels_{self.task}.pkl"
        elif self.Train_part == "Head":
            self.label_processed_dir_AF2 = f"../MyProjectData/DMEAF2{self.task}/DME{self.set_type}/labels_Head_{self.task}.pkl"
        else:
            self.label_processed_dir_AF2 = f"../MyProjectData/DMEAF2{self.task}/DME{self.set_type}/labels_Long_Tail_{self.task}.pkl"

        if self.set_type == "train":
            if self.task == "bp":
                self.flist = list(range(0, 23513))
            elif self.task == "mf":
                self.flist = list(range(0, 24956))
            else:
                self.flist = list(range(0, 11295))
        elif self.set_type == "val":
            if self.task == "bp":
                self.flist = list(range(0, 2624))
            elif self.task == "mf":
                self.flist = list(range(0, 2749))
            else:
                self.flist = list(range(0, 1299))
        else:
            self.flist = list(range(0, 3414))
        self.filelist = [os.path.join(self.processed_dir, f"{i}.bin") for i in self.flist]

        with open(self.label_processed_dir, "rb") as f:
            self.y_true = pickle.load(f)

        if self.AF2model:

            if self.set_type == "train":
                if self.task == "bp":
                    self.aflist = list(range(0, 37263))
                elif self.task == "mf":
                    self.aflist = list(range(0, 33827))
                else:
                    self.aflist = list(range(0, 32397))
            elif self.set_type == "val":
                if self.task == "bp":
                    self.aflist = list(range(0, 4146))
                elif self.task == "mf":
                    self.aflist = list(range(0, 3708))
                else:
                    self.aflist = list(range(0, 3663))
            else:
                self.aflist = list(range(0, 567))
            self.Alist = [os.path.join(self.processed_dir_AF2, f"{i}.bin") for i in self.aflist]

            with open(self.label_processed_dir_AF2, "rb") as f:
                self.y_true_AF2 = pickle.load(f)

            if (self.AF2model) and (self.set_type == "test"):
                self.filelist = self.Alist
                self.y_true = self.y_true_AF2
            else:
                self.filelist += self.Alist
                self.y_true = torch.cat((self.y_true, self.y_true_AF2), dim=0)




    def download(self):
        pass

    def process(self):
        pass

    def save(self):
        pass


    def __getitem__(self, idx):
        path = self.filelist[idx]
        graphdata, _ = dgl.load_graphs(path)

        return graphdata[0], self.y_true[idx]

    def __len__(self):
        return len(self.filelist)
