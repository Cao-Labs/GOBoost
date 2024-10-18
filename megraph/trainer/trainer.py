#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : trainer.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import os.path as osp

import numpy as np
import time
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import Subset

from dgl.dataloading import GraphDataLoader, DataLoader, ClusterGCNSampler

from ..logger import logger
from ..utils import my_tqdm


class Trainer(object):
    def __init__(
        self,
        gpu_id,
        model,
        task,
        loss_function,
        evaluator,
        meta_data,
        epochs,
        lr,
        weight_decay,
        use_adam,
        train_batch_size,
        eval_batch_size,
        num_workers,
        drop_last=False,
        record_graphs_dir=None,
        save_model_dir=None,
        graph_record_interval=10,
        stop_patience=200,
        lr_scheduler="const",
        lr_scheduler_indicator="res",
        milestones=[],
        reduce_factor=0.5,
        reduce_patience=50,
        min_lr=1e-4,
    ):
        self.gpu_id = gpu_id
        self.use_gpu = gpu_id >= 0
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.record_graphs_dir = record_graphs_dir
        self.graph_record_interval = graph_record_interval
        self.save_model_dir = save_model_dir
        self.model = model.cuda(gpu_id) if self.use_gpu else model
        self.task = task
        self.loss_function = loss_function
        self.evaluator = evaluator
        self.meta_data = meta_data
        self.stop_patience = stop_patience
        opt = Adam if use_adam else AdamW
        self.base_lr = lr
        self.optimizer = opt(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_schedule_name = lr_scheduler
        self.lr_scheduler_indicator = lr_scheduler_indicator
        self.scheduler = None
        if lr_scheduler == "plateau":
            mode = (
                "min"
                if meta_data.get("reg", False) or lr_scheduler_indicator == "loss"
                else "max"
            )
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=reduce_factor,
                patience=reduce_patience,
                verbose=True,
                min_lr=min_lr,
            )
        elif lr_scheduler == "multistep":
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=reduce_factor,
            )

    def get_graph_and_labels(self, item):
        if self.task == "gpred":
            g, labels = item
        else:
            g = item
            labels = g.ndata["label"]
        # process labels
        if self.meta_data["loss_func"] in ["BCE", "MSE", "L1"]:
            labels = labels.to(torch.float32)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(-1)
        elif self.meta_data["loss_func"] in ["CE", "CB", "WCE"]:
            labels = labels.squeeze(-1)
            labels = labels.to(torch.int64)
        return g, labels

    def train_epoch(self, loaders, masks):
        train_loader = loaders[0]  # 只使用了 train 数据加载器
        train_mask = None if self.meta_data["inductive"] else masks[0]
        # 我们自己的任务是"inductive"，这里train_mask = None
        self.model.train()
        all_logits, all_labels, all_loss = [], [], []
        for i, item in enumerate(my_tqdm(train_loader, desc="Iteration")):
            # 对训练数据加载器进行迭代。my_tqdm是一个进度条函数。
            g, labels = self.get_graph_and_labels(item)
            if self.meta_data["inductive"] and self.use_gpu:
                g = g.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
            logits = self.model(g)  # 直接将图数据放入模型中
            mask = (
                g.ndata["train_mask"].bool()
                if self.meta_data["subg_sampling"]
                else train_mask
            )
            # 在"inductive"任务中这里应该是mask = train_mask = None
            loss = self.get_loss(logits, labels, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_logits.append(logits)
            all_labels.append(labels)
            all_loss.append(loss.item())
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        train_result = self.get_res(logits, labels, mask=train_mask)
        return train_result, np.mean(all_loss)

    def get_res(self, logits, labels, mask=None):
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(-1)
        y_pred = logits

        if self.meta_data.get("eval_metric", None) == "SBM_acc":
            logits = torch.nn.Softmax(dim=1)(logits)

        if self.meta_data["loss_func"] in ["CE", "WCE"]:
            y_pred = torch.argmax(logits, dim=1, keepdim=True)

        if mask is not None:
            y_pred, labels = y_pred[mask], labels[mask]
        input_dict = dict(y_pred=y_pred, y_true=labels)
        return self.evaluator.eval(input_dict)

    def get_loss(self, logits, labels, mask=None):
        if mask is not None:
            logits, labels = logits[mask], labels[mask]
        is_labeled = labels == labels  # ignore Nan value in labels
        loss = self.loss_function(logits[is_labeled], labels[is_labeled])
        return loss

    def evaluate(self, loaders, masks):
        eval_results = []
        loss_results = []
        self.model.eval()

        def get_logits(g, is_first=False):
            if (
                self.record_graphs_dir is not None
                and is_first
                and self.cur_epoch_id % self.graph_record_interval == 0
            ):  # only the first one
                fname = osp.join(
                    self.record_graphs_dir, f"graphs_{self.cur_epoch_id:04d}"
                )
                return self.model(g, save_graphs_filename=fname)
            else:
                return self.model(g)

        debug = False
        with torch.no_grad():
            if self.meta_data["inductive"]:
                for i in range(1, 3):
                    all_logits, all_labels = [], []
                    for j, item in enumerate(loaders[i]):
                        g, labels = self.get_graph_and_labels(item)
                        if self.use_gpu:
                            g = g.to(self.gpu_id)
                            labels = labels.to(self.gpu_id)
                        # only save the first one of val (when needed)
                        logits = get_logits(g, is_first=(i == 1 and j == 0))
                        if debug:  # DEBUG ONLY
                            for name, param in self.model.named_parameters():
                                if param.requires_grad:
                                    logger.info(f"{name}, {param.data}")
                            logger.info("-" * 30 + " labels " + "-" * 30)
                            logger.info(labels)
                        all_logits.append(logits)
                        all_labels.append(labels)
                    logits = torch.cat(all_logits, dim=0)
                    labels = torch.cat(all_labels, dim=0)
                    eval_results.append(self.get_res(logits, labels))
                    loss_results.append(self.get_loss(logits, labels).detach().item())
            else:
                all_logits, all_labels = [], []
                all_val_masks, all_test_masks = [], []
                for i, item in enumerate(loaders[1]):
                    g, labels = self.get_graph_and_labels(item)
                    logits = get_logits(g, is_first=True)
                    all_logits.append(logits)
                    all_labels.append(labels)
                    all_val_masks.append(g.ndata["val_mask"])
                    all_test_masks.append(g.ndata["test_mask"])
                logits = torch.cat(all_logits, dim=0)
                labels = torch.cat(all_labels, dim=0)
                val_masks = torch.cat(all_val_masks, dim=0).bool()
                test_masks = torch.cat(all_test_masks, dim=0).bool()
                masks = (
                    [val_masks, test_masks]
                    if self.meta_data["subg_sampling"]
                    else masks[1:]
                )
                for mask in masks:
                    eval_results.append(self.get_res(logits, labels, mask))
                    loss_results.append(
                        self.get_loss(logits, labels, mask).detach().item()
                    )
        return eval_results, loss_results

    def get_loaders(self, dataset, masks=None):
        if self.meta_data["inductive"]:
            loaders = []
            for ind, mask in enumerate(masks):
                is_train = ind == 0
                batch_size = self.train_batch_size if is_train else self.eval_batch_size
                if mask.dtype == torch.bool:
                    mask = torch.arange(len(mask))[mask]
                    # mask 就从一个布尔类型的张量变成了一个包含了原张量中所有 True 对应位置的索引的整数类型张量。
                loaders.append(
                    GraphDataLoader(
                        Subset(dataset, mask),
                        batch_size=batch_size,
                        shuffle=is_train,
                        num_workers=self.num_workers,
                        drop_last=self.drop_last,
                    )
                )
                # 这里应该是加载了train val test 的三个数据加载器
            return loaders
        else:
            # Transductive learning on a single graph
            graph = dataset[0]
            if not self.meta_data["subg_sampling"]:
                if self.use_gpu:
                    graph = graph.to(self.gpu_id)
                return [[graph], [graph]]
            else:
                device = f"cuda:{self.gpu_id}" if self.use_gpu else "cpu"
                cache_path = self.meta_data.get("cache_path", "cluster.pkl")
                num_parts = self.meta_data.get("subg_num", 1000)
                sampler = ClusterGCNSampler(
                    graph,
                    num_parts,
                    cache_path=cache_path,
                    prefetch_ndata=[
                        "feat",
                        "label",
                        "train_mask",
                        "val_mask",
                        "test_mask",
                    ],
                )
                train_dataloader = DataLoader(
                    graph,
                    torch.arange(num_parts).to(device),
                    sampler,
                    device=device,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=self.num_workers,
                    use_uva=self.use_gpu,
                )
                eval_dataloader = DataLoader(
                    graph,
                    torch.arange(num_parts).to(device),
                    sampler,
                    device=device,
                    batch_size=self.eval_batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.num_workers,
                    use_uva=self.use_gpu,
                )
                return [train_dataloader, eval_dataloader]

    def train(self, epochs, run_id, dataset, masks):
        loaders = self.get_loaders(dataset, masks)  # 获得train val test 数据集加载器
        if self.gpu_id >= 0:
            masks = [mask.to(self.gpu_id) for mask in masks]

        key = self.meta_data.get("eval_metric", "acc")  # 评价指标

        def is_better(x, y):  # 根据不同的评价指标，去比较结果的好坏
            if y is None:
                return True
            if key in ["L2", "MAE", "rmse"]:
                return x <= y
            return x >= y

        dur = []
        best_val_res = None
        final_test_res = None
        kept_epochs = 0
        for epoch in range(epochs):
            record_time = epoch >= 3
            # 当epoch大于或等于3时，record_time将被设置为True，否则为False。
            t0 = time.time()
            self.cur_epoch_id = epoch
            train_result, loss = self.train_epoch(loaders, masks)
            eval_results, loss_results = self.evaluate(loaders, masks)
            val_result, test_result = eval_results
            val_loss, test_loss = loss_results

            train_res = train_result[key]
            val_res = val_result[key]
            test_res = test_result[key]

            cur_lr = self.base_lr
            if self.scheduler is not None:
                if self.lr_schedule_name == "plateau":
                    if self.lr_scheduler_indicator == "res":
                        self.scheduler.step(val_res)
                    elif self.lr_scheduler_indicator == "loss":
                        self.scheduler.step(val_loss)
                    cur_lr = np.mean(self.scheduler._last_lr)
                else:
                    self.scheduler.step()
                    cur_lr = np.mean(self.scheduler.get_last_lr())

            if is_better(val_res, best_val_res):
                best_val_res = val_res
                final_test_res = test_res
                kept_epochs = 0
                if self.save_model_dir is not None:
                    self.save(epoch, run_id)
            else:
                kept_epochs += 1
            if record_time:
                dur.append(time.time() - t0)
            avg_time = np.mean(dur) if len(dur) > 0 else 0.0
            logger.record("run", run_id)
            logger.record("epoch", epoch)
            logger.record("time_s", avg_time)
            logger.record("train_loss", loss)
            logger.record("val_loss", val_loss)
            logger.record("test_loss", test_loss)
            logger.record("train_res", train_res)
            logger.record("val_res", val_res)
            logger.record("test_res", test_res)
            logger.record("best_val_res", best_val_res)
            logger.record("final_test_res", final_test_res)
            logger.record("kept_epochs", kept_epochs)
            logger.record("lr", cur_lr)
            logger.dump(epoch)
            if self.stop_patience is not None and kept_epochs >= self.stop_patience:
                # early_stop
                break
        return final_test_res

    def save(self, epoch, run_id):
        torch.save(
            self.model.state_dict(),
            osp.join(self.save_model_dir, f"model_{run_id:02d}.pth"),
        )
        logger.info(f"Save best model in epoch {epoch}.")

    def load(self, model_path):
        device = f"cuda:{self.gpu_id}" if self.use_gpu else "cpu"
        model_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(model_dict)
        logger.info(f"Load model from {model_path}.")

    __hyperparams__ = []
    __parser__ = None

    @classmethod
    def _set_parser(cls, parser):
        cls.__parser__ = parser

    @classmethod
    def _add_argument(cls, name, *args, **kwargs):
        cls.__hyperparams__.append(name)
        name = name.replace("_", "-")
        cls.__parser__.add_argument("--" + name, *args, **kwargs)

    @classmethod
    def register_trainer_args(cls, parser):
        cls._set_parser(parser.add_argument_group("trainer"))
        cls._add_argument(
            "epochs", "-ep", type=int, default=200, help="number of epochs"
        )
        cls._add_argument("lr", "-lr", type=float, default=1e-2, help="learning rate")
        cls._add_argument(
            "weight_decay", "-wd", type=float, default=5e-4, help="weight decay"
        )
        cls._add_argument(
            "use_adam", "-ad", action="store_true", help="Use Adam instead of AdamW"
        )
        cls._add_argument(
            "train_batch_size", "-tbs", type=int, default=32, help="Train batch size"
        )
        cls._add_argument(
            "eval_batch_size", "-ebs", type=int, default=32, help="Eval batch size"
        )
        cls._add_argument(
            "num_workers", "-nw", type=int, default=0, help="Number of workers"
        )
        cls._add_argument(
            "drop_last", "-dpl", action="store_true", help="Drop last batch"
        )
        cls._add_argument(
            "stop_patience",
            "-spt",
            type=int,
            default=None,
            help="Patience for early stop",
        )
        cls._add_argument(
            "lr_scheduler",
            "-lrs",
            choices=["const", "multistep", "plateau"],
            default="const",
            help="lr scheduler",
        )
        cls._add_argument(
            "lr_scheduler_indicator",
            "-lrsi",
            choices=["loss", "res"],
            default="res",
            help="lr scheduler indicator",
        )
        cls._add_argument(
            "milestones",
            "-mst",
            type=int,
            nargs="+",
            default=[],
            help="Milestones for multistep LR scheduler",
        )
        cls._add_argument(
            "reduce_factor",
            "-rf",
            type=float,
            default=0.5,
            help="Reduce factor for scheduler",
        )
        cls._add_argument(
            "reduce_patience",
            "-rpt",
            type=int,
            default=50,
            help="Patience for scheduler",
        )
        cls._add_argument(
            "min_lr", "-mlr", type=float, default=1e-4, help="Minimal Lr for scheduler"
        )
        cls._add_argument(
            "graph_record_interval",
            "-gri",
            type=int,
            default=10,
            help="The save interval of recording graphs",
        )

    @classmethod
    # cls 是一个类对象。这是一个常见的命名约定，用于表示类本身。在这个上下文中，cls 是一个类方法的第一个参数，它代表了调用该方法的类。
    def from_args(cls, args, **kwargs):
        init_params = {k: getattr(args, k) for k in cls.__hyperparams__}
        init_params.update(kwargs)
        '''
        如果 kwargs 中的键（即参数名）在 init_params 中已经存在，那么它会更新 init_params 中对应键的值为 kwargs 中的值。
        如果 kwargs 中的键在 init_params 中不存在，那么它会在 init_params 中添加这个键，并设置其值为 kwargs 中的值。
        '''
        return cls(**init_params)
        # 这个函数返回的是一个实例化类的对象，实例的参数就是init_params
