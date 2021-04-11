import sys
import time
import torch
import traceback
import numpy as np
import torch.nn.functional as F
from graphnas.gnn import GraphNet
from graphnas.utils.model_utils import calculate_acc
from graphnas.utils.model_utils import process_action
from graphnas.utils.model_constants import epoch_info_str
from graphnas.utils.model_constants import optimization_metrics


# manager the train process of GNN on citation dataset
class CitationGNNManager(object):
    def __init__(self, args):
        self.args = args
        print('args on CitationGNNManager:', args)
        self.args = args
        self.drop_out = args.in_drop
        self.multi_label = args.multi_label
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.retrain_epochs = args.retrain_epochs
        self.epochs = args.epochs
        self.train_graph_index = 0
        self.train_set_length = 10
        self.param_file = args.param_file
        self.shared_params = None
        self.loss_fn = torch.nn.functional.nll_loss
        self.initialize_log_file()

    def load_param(self):
        # don't share param
        pass

    def save_param(self, model, update_all=False):
        # don't share param
        pass

    # train from scratch
    def evaluate(self, actions=None, format="two"):
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)

        if self.args.cuda:
            model.cuda()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
        try:
            model, val_acc, test_acc = self.run_model(model, optimizer,
                                                      self.loss_fn,
                                                      self.data, self.epochs,
                                                      cuda=self.args.cuda,
                                                      return_best=True)
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
                test_acc = 0
            else:
                raise e
        return val_acc, test_acc

    # train from scratch
    def train(self, actions=None, format="two"):
        origin_action = actions
        actions = process_action(actions, format, self.args)
        print("train action:", actions)

        # create model
        model = self.build_gnn(actions)
        val_acc, metrics = {}, {}
        try:
            if self.args.cuda:
                model.cuda()
            # use optimizer
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
            if self.args.batches > 1:
                model, val_acc, metrics = \
                    self.run_batched_model(model, optimizer,
                                           self.loss_fn, self.data,
                                           self.epochs, cuda=self.args.cuda,
                                           n_splits=self.args.batches)
            else:
                if self.args.folds > 1:
                    model, val_acc, metrics = \
                        self.run_k_fold_model(model, optimizer,
                                              self.loss_fn, self.data,
                                              self.epochs, cuda=self.args.cuda,
                                              seed=self.args.random_seed,
                                              n_splits=self.args.folds)
                else:
                    model, val_acc, metrics = \
                        self.run_model(model,
                                       optimizer,
                                       self.loss_fn,
                                       self.data,
                                       self.epochs,
                                       cuda=self.args.cuda,
                                       # , show_info=True
                                       )
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                val_acc = 0
                metrics = {metric: 0.0 for metric in optimization_metrics}
            print(e, repr(e))
            traceback.print_exc(file=sys.stdout)
        # Log the current architecture with metrics
        self.record_action_info(origin_action, metrics)
        return val_acc, metrics

    @property
    def record_action_info_filename(self):
        return "_".join([self.args.dataset,
                         self.args.search_mode,
                         self.args.optimizer,
                         self.args.opt_metric,
                         str(self.args.random_seed),
                         self.args.log_file])

    def initialize_log_file(self):
        with open(self.record_action_info_filename, 'w') as f:
            f.write("arch;")
            f.write(";".join(optimization_metrics))
            f.write("\n")

    def record_action_info(self, origin_action, metrics):
        with open(self.record_action_info_filename, "a") as f:
            f.write(str(origin_action))
            f.write(";")
            # Write arch. metrics in the order they appear in the list
            f.write(';'.join([str(metrics[metric_name])
                              for metric_name in optimization_metrics]))
            f.write("\n")

    def build_gnn(self, actions):
        model = GraphNet(actions, self.in_feats, self.n_classes,
                         drop_out=self.args.in_drop, multi_label=False,
                         batch_normal=False)
        return model

    def retrain(self, actions, format="two"):
        return self.train(actions, format)

    @staticmethod
    def run_model(model, optimizer, loss_fn, data, epochs,
                  early_stop=5, tmp_model_file="geo_citation.pkl",
                  half_stop_score=0, return_best=False,
                  cuda=True, need_early_stop=False, show_info=False):
        print('chamou o run_model da CitationGNNManager')
        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")
        model_val_acc = 0
        features, g, labels, mask, val_mask, test_mask, n_edges = \
            CitationGNNManager.prepare_data(data, cuda)

        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            # forward
            logits = model(features, g)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[mask], labels[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            # evaluate
            model.eval()
            logits = model(features, g)
            logits = F.log_softmax(logits, 1)
            # Get outputs
            _, output = torch.max(logits, dim=1)
            # Calculate accuracy
            train_acc = calculate_acc(output, labels, mask)
            dur.append(time.time() - t0)
            val_loss = float(loss_fn(logits[val_mask], labels[val_mask]))
            val_acc = calculate_acc(output, labels, val_mask)
            test_acc = calculate_acc(output, labels, test_mask)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                model_val_acc = val_acc
                if test_acc > best_performance:
                    best_performance = test_acc
            if show_info:
                print(epoch_info_str.format(epoch,
                                            loss.item(),
                                            np.mean(dur),
                                            train_acc,
                                            val_acc,
                                            test_acc))
                end_time = time.time()
                print("Each Epoch Cost Time: %f " %
                      ((end_time - begin_time) / epoch))
        print("val_score:" + str(model_val_acc),
              "test_score:" + str(best_performance))
        if return_best:
            return model, model_val_acc, best_performance
        else:
            return model, model_val_acc

    @staticmethod
    def run_k_fold_model(model, optimizer, loss_fn, data, epochs,
                         seed=10, n_splits=10,
                         early_stop=5, tmp_model_file="geo_citation.pkl",
                         half_stop_score=0, return_best=False,
                         cuda=True, need_early_stop=False, show_info=False):
        pass
