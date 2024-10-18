from NEWDGLDataSet import GoTermDataset, collate_fn
from utils import log
from sklearn import metrics
from megraph.utils import my_tqdm
from Loss.MultiGrainedFocalLoss import *
from Loss.TrainLossVal import *
from Loss.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from evaluation_matrics import *

import os
import sys
import torch

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

parser.add_argument("--AF2model", "-AF2m", default=False, type=bool, help='whether to use AF2model for training')
parser.add_argument("--batch_size", "-bs", type=int, default=16, help='')
parser.add_argument("--epochs", "-ep", type=int, default=200, help="number of epochs")
parser.add_argument("--sub_function", "-sf", type=str, default="bp", help="A subclass of protein function")
parser.add_argument("--isTrain", "-isT", default=False, type=bool, help='whether to retrain')


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
parser.add_argument("--gpu-id", "-gid", type=int, default=0, help="gpu id")
parser.add_argument("--dataset_name", "-dname", type=str, default="protein", help="The input dataset")
parser.add_argument("--dataset_subname", "--dsub", type=str, default=None, help="The name for the sub dataset, if applicable")
parser.add_argument("--train_part", "-tpt", type=str, default="All", help="Specific basic model")


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
    # Load GO annotations
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

def run(run_id):
    train_set = GoTermDataset(set_type="train", task=args.sub_function, AF2model=args.AF2model, Train_part=args.train_part)
    valid_set = GoTermDataset(set_type="val", task=args.sub_function, AF2model=args.AF2model, Train_part=args.train_part)
    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, drop_last=False, num_workers=4, prefetch_factor=2)
    val_loader = GraphDataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, drop_last=False, num_workers=4, prefetch_factor=2)
    output_dim = valid_set.y_true.shape[-1]
    task = 'gpred'

    input_dims = [0, 512, 0]
    output_dims = [1943, 0, 0]
    output_dims[0] = output_dim
    pe_dim = 0

    if args.layer in ["gcn"]:
        input_dims[2] = 0
    # create graph model
    if args.model in ["plain", "megraph", "unet", "hgnet"]:

        def build_conv(**kwargs):
            return graph_layer.from_args(args, **kwargs)

        model = graph_model.from_args(
            args,
            input_dims=input_dims,
            output_dims=output_dims,
            pe_dim=pe_dim,
            task=task,
            embed_method={},
            build_conv=build_conv,
        )
    else:
        in_feats = input_dims[1]
        n_classes = graph_dataset_manager.get_num_classes()
        model = graph_model.from_args(args, in_feats=in_feats, n_classes=n_classes)

    model = model.to(args.device)
    logger.info(model)
    logger.info(f"Num params of {args.model}: {get_num_params(model)}")


    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **args.optimizer,
    )

    start_epoch = 0
    bce_loss = create_loss(gamma_neg=args.co, pow=args.pow,)
    if args.AF2model:
        if args.train_part == "All":
            distribution_path = f"./data/DMETrain/distribution_{args.sub_function}_All.txt"
        elif args.train_part == "Head":
            distribution_path = f"./data/DMETrain/distribution_Head_{args.sub_function}_All.txt"
        else:
            distribution_path = f"./data/DMETrain/distribution_Tail_{args.sub_function}_All.txt"
    else:
        if args.train_part == "All":
            distribution_path = f"./data/DMETrain/distribution_{args.sub_function}.txt"
        elif args.train_part == "Head":
            distribution_path = f"./data/DMETrain/distribution_Head_{args.sub_function}.txt"
        else:
            distribution_path = f"./data/DMETrain/distribution_Tail_{args.sub_function}.txt"

    bce_loss.create_weight(distribution_path)
    bce_loss_val = train_create_loss(gamma_neg=args.co, pow=args.pow,)
    bce_loss_val.create_weight(distribution_path)

    prot2annot, goterms, gonames, counts = load_GO_annot('data/nrSwiss-Model-GO_annot.tsv')
    processed_dir = '../MyProjectData/DMEDATA'
    pdbch_list = torch.load(os.path.join(processed_dir, f"val_pdbch.pt"))[f"val_pdbch"]
    pdbch_af2_list = torch.load(os.path.join(processed_dir, f"AF2val_pdbch.pt"))[f"val_pdbch"]
    pdbch_list = pdbch_list + pdbch_af2_list

    val_loss = []
    es = 0
    best_inx = -1
    y_true_all = valid_set.y_true.float().reshape(-1)

    modeldirone = f"/mnt/c/{args.train_part}-{args.sub_function}/"
    modeldirtwo = f"/mnt/c/{args.train_part}-{args.sub_function}/PredictionPkl/"
    modeldirthree = f"/mnt/c/{args.train_part}-{args.sub_function}/PredictionPklVal/"

    if not os.path.exists(modeldirone):
        os.mkdir(modeldirone)
    if not os.path.exists(modeldirtwo):
        os.mkdir(modeldirtwo)
    if not os.path.exists(modeldirthree):
        os.mkdir(modeldirthree)

    for ith_epoch in range(start_epoch, args.epochs):
        model.perturbed = False
        model.set_perturbed(False)
        for idx_batch, batch in enumerate(my_tqdm(train_loader, desc="Iteration")):
            model.train()
            out1, out2 = model(batch[0].to(args.device))
            y_true = batch[1].to(args.device)
            mgf_loss = bce_loss(out1, out2, y_true)
            loss = mgf_loss
            if idx_batch % 200 == 0:
                log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss), 6)}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        y_pred_all = []

        with torch.no_grad():
            model.perturbed = False
            model.set_perturbed(False)
            for idx_batch, batch in enumerate(val_loader):
                out1, out2 = model(batch[0].to(args.device))
                out = 0.5*out1 + 0.5*out2
                out = torch.sigmoid(out)
                y_pred_all.append(out)

            y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1)
            y_pred_all_2 = y_pred_all.reshape(int(len(y_true_all) / output_dim), output_dim)
            all_val_loss = bce_loss_val(y_pred_all_2.cpu(), valid_set.y_true.cpu())
            log(f"{ith_epoch} VAL_epoch ||| loss: {round(float(all_val_loss), 6)}")

            val_loss.append(all_val_loss.numpy())

            if ith_epoch == start_epoch:
                best_eval_loss = all_val_loss

            if all_val_loss < best_eval_loss:
                best_eval_loss = all_val_loss
                es = 0
                best_inx = ith_epoch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
                }, modeldirone + task + f"best_{args.sub_function}_{args.train_part}.pt")
            else:
                es += 1
                print("Loss Counter {} of 10".format(es))

            torch.save(
                {
                    "val_bce": val_loss,
                }, modeldirone + task + f"{args.train_part}-{args.sub_function}_ValLoss.pt"
            )

            if (es > 9):
                print(f'best_inx={best_inx}')
                break
    return


def main():
    for i in range(args.runs):
        run(i)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_info = traceback.format_exc()
        logger.error(f"\n{error_info}")

    # python train.py --train_part All --batch_size 32 --epochs 100 --sub_function cc --AF2model True --config-file ./configs/best/cfg_protein_cc.py
    # python train.py --train_part All --batch_size 32 --epochs 100 --sub_function cc --config-file ./configs/best/cfg_protein_cc.py