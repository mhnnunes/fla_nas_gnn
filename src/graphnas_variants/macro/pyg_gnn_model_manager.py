import time
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
# Utils
from graphnas.utils.data_utils import load_data
from graphnas.utils.model_utils import calculate_acc
from graphnas.utils.model_constants import epoch_info_str
from graphnas.utils.data_utils import split_and_batch_data
from graphnas.utils.model_utils import calculate_prec_recall_fscore

from graphnas.gnn_model_manager import CitationGNNManager

from graphnas_variants.macro.pyg_gnn import GraphNet


def update_val_metrics_dict(epoch_metrics, val_loss, output, y, val_, t0):
    val_acc = calculate_acc(output, y, val_)
    (prec_macro, recall_macro, f_macro,
     prec_micro, recall_micro, f_micro,
     prec_weigh, recall_weigh, f_weigh) = \
        calculate_prec_recall_fscore(output, y, val_)
    # Update epoch metrics dict
    epoch_metrics['val_loss'] += val_loss
    epoch_metrics['val_acc'] += val_acc
    epoch_metrics['prec_macro'] += prec_macro
    epoch_metrics['prec_micro'] += prec_micro
    epoch_metrics['prec_weigh'] += prec_weigh
    epoch_metrics['recall_macro'] += recall_macro
    epoch_metrics['recall_micro'] += recall_micro
    epoch_metrics['recall_weigh'] += recall_weigh
    epoch_metrics['f_macro'] += f_macro
    epoch_metrics['f_micro'] += f_micro
    epoch_metrics['f_weigh'] += f_weigh
    epoch_metrics['avg_epoch_time'] += time.time() - t0


class GeoCitationManager(CitationGNNManager):
    def __init__(self, args):
        super(GeoCitationManager, self).__init__(args)
        print("on GeoCitationManager, args: ", args)
        self.data = load_data(args.dataset,
                              self.args.random_seed,
                              self.args.folds)
        # print("on GeoCitationManager, data: ", self.data)
        self.args.in_feats = self.in_feats = self.data.num_features
        self.args.num_class = self.n_classes = self.data.y.max().item() + 1
        print('in_feats: ', self.in_feats, '  n_classes: ', self.n_classes)

    def build_gnn(self, actions):
        # CHAMA ESSE
        model = GraphNet(actions, self.in_feats, self.n_classes,
                         drop_out=self.args.in_drop, multi_label=False,
                         batch_normal=False, residual=False)
        return model

    def update_args(self, args):
        self.args = args

    def save_param(self, model, update_all=False):
        pass

    @staticmethod
    def run_model(model, optimizer, loss_fn, data, epochs,
                  early_stop=5, tmp_model_file="geo_citation.pkl",
                  half_stop_score=0, return_best=False,
                  cuda=True, need_early_stop=False, show_info=False):
        # print('chamou o run_model da GeoCitationManager')
        dur = []
        begin_time = time.time()
        min_val_loss = float("inf")
        model_val_acc = 0
        # Load full data to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data.to(device)
        model.train()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            # forward
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _ = loss.item()
            # evaluate
            model.eval()
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            # Get model output
            _, output = torch.max(logits, dim=1)
            # Calculate metrics
            epoch_metrics = defaultdict(float)
            epoch_metrics['train_acc'] = \
                calculate_acc(output, data.y, data.train_mask)
            dur.append(time.time() - t0)
            # Update validation metrics
            val_loss = loss_fn(logits[data.valid_mask],
                               data.y[data.valid_mask])
            epoch_metrics['val_loss'] = val_loss.item()
            update_val_metrics_dict(epoch_metrics, val_loss,
                                    output, data.y, data.valid_mask, t0)
            if epoch_metrics['val_loss'] < min_val_loss:
                min_val_loss = epoch_metrics['val_loss']
                model_val_acc = epoch_metrics['val_acc']
            if show_info:
                print(epoch_info_str.format(epoch,
                                            loss.item(),
                                            np.mean(dur),
                                            epoch_metrics['train_acc'],
                                            epoch_metrics['val_acc']))

                end_time = time.time()
                print("Each Epoch Cost Time: %f " %
                      ((end_time - begin_time) / epoch))
        print("val_score:", str(model_val_acc))
        return model, model_val_acc, epoch_metrics

    @staticmethod
    def run_k_fold_model(model, optimizer, loss_fn, data, epochs,
                         seed=10, n_splits=10, early_stop=5,
                         tmp_model_file="geo_citation.pkl",
                         half_stop_score=0, return_best=False,
                         cuda=True, need_early_stop=False, show_info=False):
        begin_time = time.time()
        min_val_loss = float("inf")
        model_val_acc = 0
        # Load full data to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data.to(device)
        # Start training
        model.train()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            # Train and validate on each fold
            epoch_metrics = defaultdict(float)
            for i in range(data.train_folds.shape[1]):
                train_, val_ = data.train_folds[:, i], data.valid_folds[:, i]
                # forward
                logits = model(data.x, data.edge_index)
                logits = F.log_softmax(logits, 1)
                train_loss = loss_fn(logits[train_], data.y[train_])
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                train_loss = train_loss.item()
                # Calculate accuracy on train dataset
                _, output = torch.max(logits, dim=1)
                train_acc = calculate_acc(output, data.y, train_)
                # Accumulate metrics on dict
                epoch_metrics['train_loss'] += train_loss
                epoch_metrics['train_acc'] += train_acc
                # Evaluate model on validation set (of the current fold)
                with torch.no_grad():
                    model.eval()
                    logits = model(data.x, data.edge_index)
                    logits = F.log_softmax(logits, 1)
                    val_loss = loss_fn(logits[val_],
                                       data.y[val_])
                    val_loss = val_loss.item()
                    # Calculate metrics on validation dataset
                    _, output = torch.max(logits, dim=1)
                    update_val_metrics_dict(epoch_metrics, val_loss, output,
                                            data.y, val_, t0)
            # Normalize metrics over folds
            epoch_metrics = {k: (v / n_splits)
                             for k, v in epoch_metrics.items()}
            epoch_metrics['train_time'] = time.time() - begin_time
            if epoch_metrics['val_loss'] < min_val_loss:
                min_val_loss = epoch_metrics['val_loss']
                model_val_acc = epoch_metrics['val_acc']
            # End of Cross validation, print model stats if necessary
            if show_info:
                print(epoch_info_str.format(epoch,
                                            epoch_metrics['val_loss'],
                                            time.time() - t0,
                                            epoch_metrics['train_acc'],
                                            epoch_metrics['val_acc'],
                                            0.0))
                end_time = time.time()
                print("Each Epoch Cost Time: %f " %
                      ((end_time - begin_time) / epoch))
        print("val_score:", str(model_val_acc))
        print(epoch_metrics)
        return model, model_val_acc, epoch_metrics

    @staticmethod
    def run_batched_model(model, optimizer, loss_fn, data, epochs,
                          early_stop=5, tmp_model_file="geo_citation.pkl",
                          half_stop_score=0, return_best=False, n_splits=10,
                          cuda=True, need_early_stop=False, show_info=False):
        dur = []
        begin_time = time.time()
        min_val_loss = float("inf")
        model_val_acc = 0
        # Get batches
        train_loader = split_and_batch_data(
            data, batches=n_splits)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.train()
        for epoch in range(1, epochs + 1):
            epoch_metrics = defaultdict(float)
            t0 = time.time()
            for data in train_loader:
                data.to(device)
                # forward
                logits = model(data.x, data.edge_index)
                logits = F.log_softmax(logits, 1)
                loss = loss_fn(logits[data.train_mask],
                               data.y[data.train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _ = loss.item()
                # evaluate
                model.eval()
                logits = model(data.x, data.edge_index)
                logits = F.log_softmax(logits, 1)
                # Get model output
                _, output = torch.max(logits, dim=1)
                # Calculate accuracy
                epoch_metrics['train_acc'] += \
                    calculate_acc(output, data.y, data['train_mask'])
                dur.append(time.time() - t0)
                # Evaluate model on validation set (of the current BATCH)
                with torch.no_grad():
                    model.eval()
                    logits = model(data.x, data.edge_index)
                    logits = F.log_softmax(logits, 1)
                    val_loss = loss_fn(logits[data['valid_mask']],
                                       data.y[data['valid_mask']])
                    val_loss = val_loss.item()
                    # Calculate metrics on validation dataset
                    _, output = torch.max(logits, dim=1)
                    update_val_metrics_dict(epoch_metrics, val_loss, output,
                                            data.y, data['valid_mask'], t0)
            epoch_metrics = {k: (v / n_splits)
                             for k, v in epoch_metrics.items()}
            epoch_metrics['train_time'] = time.time() - begin_time
            if epoch_metrics['val_loss'] < min_val_loss:
                min_val_loss = epoch_metrics['val_loss']
                model_val_acc = epoch_metrics['val_acc']
            if show_info:
                print(epoch_info_str.format(epoch,
                                            loss.item(),
                                            np.mean(dur),
                                            epoch_metrics['train_acc'],
                                            epoch_metrics['val_acc'],
                                            0.0))

                end_time = time.time()
                print("Each Epoch Cost Time: %f " %
                      ((end_time - begin_time) / epoch))
        print("val_score:", str(model_val_acc))
        return model, model_val_acc, epoch_metrics
