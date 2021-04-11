import time
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score
# GNN Managers
from graphnas_variants.macro.pyg_gnn import GraphNet
from graphnas_variants.micro.micro_gnn import MicroGNN
from graphnas.utils.model_constants import epoch_info_str
from graphnas.utils.data_utils import split_and_batch_data
from sklearn.metrics import precision_recall_fscore_support

# Loss is common to all child GNNs
loss_fn = F.nll_loss  # Negative Log-Likelihood Loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FixedList(list):
    def __init__(self, size=10):
        super(FixedList, self).__init__()
        self.size = size

    def append(self, obj):
        if len(self) >= self.size:
            self.pop(0)
        super().append(obj)


class TopAverage(object):
    def __init__(self, top_k=10):
        self.scores = []
        self.top_k = top_k

    def get_top_average(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return 0

    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0
        # print("Top %d average: %f" % (self.top_k, avg))
        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores = self.scores[:self.top_k]
        return avg

    def get_reward(self, score):
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)


class EarlyStop(object):
    def __init__(self, size=10):
        self.size = size
        self.train_loss_list = FixedList(size)
        self.train_score_list = FixedList(size)
        self.val_loss_list = FixedList(size)
        self.val_score_list = FixedList(size)

    def should_stop(self, train_loss, train_score, val_loss, val_score):
        flag = False
        if len(self.train_loss_list) < self.size:
            pass
        else:
            if val_loss > 0:
                # and val_score <= np.mean(self.val_score_list)
                if val_loss >= np.mean(self.val_loss_list):
                    flag = True
            elif train_loss > np.mean(self.train_loss_list):
                flag = True
        self.train_loss_list.append(train_loss)
        self.train_score_list.append(train_score)
        self.val_loss_list.append(val_loss)
        self.val_score_list.append(val_score)

        return flag

    def should_save(self, train_loss, train_score, val_loss, val_score):
        if len(self.val_loss_list) < 1:
            return False
        minimum_train_loss = (train_loss < min(self.train_loss))
        minimum_val_score = (val_score > max(self.val_score_list))
        if minimum_train_loss and minimum_val_score:
            return True
        else:
            return False


def process_action(actions, type, args):
    if type == 'two':
        actual_action = actions
        # actual_action[-2] = args.num_class
        actual_action[-1] = args.num_class
        # actual_action.append( args.num_class)
        return actual_action
    elif type == "simple":
        actual_action = actions
        index = len(actual_action) - 1
        actual_action[index]["out_dim"] = args.num_class
        return actual_action
    elif type == "dict":
        return actions
    elif type == "micro":
        return actions


def calculate_acc(y_pred, y_true, mask):
    correct = torch.sum(y_pred[mask] == y_true[mask])
    # Return accuracy: % of correctly guessed labels
    return correct.item() * 1.0 / mask.sum().item()


def calc_f1(output, labels, sigmoid=True):
    y_true = labels.cpu().data.numpy()
    y_pred = output.cpu().data.numpy()
    if not sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return (f1_score(y_true, y_pred, average="micro"),
            f1_score(y_true, y_pred, average="macro"))


def calculate_prec_recall_fscore(model_output, y, mask):
    yval_cpu = y[mask].cpu().numpy()
    ypred_cpu = model_output[mask].cpu().numpy()
    prec_macro, recall_macro, f_macro, _ = \
        precision_recall_fscore_support(
            yval_cpu,
            ypred_cpu,
            average='macro',
            zero_division=0)
    prec_micro, recall_micro, f_micro, _ = \
        precision_recall_fscore_support(
            yval_cpu,
            ypred_cpu,
            average='micro',
            zero_division=0)
    prec_weigh, recall_weigh, f_weigh, _ = \
        precision_recall_fscore_support(
            yval_cpu,
            ypred_cpu,
            average='weighted',
            zero_division=0)
    return (prec_macro, recall_macro, f_macro,
            prec_micro, recall_micro, f_micro,
            prec_weigh, recall_weigh, f_weigh)


def construct_actions(actions, action_list, search_space):
    structure_list = []
    for single_action in actions:
        structure = []
        print('single_action: ', single_action)
        for action, action_name in zip(single_action, action_list):
            predicted_actions = search_space[action_name][action]
            structure.append(predicted_actions)
        structure_list.append(structure)
    return structure_list


def update_metrics_dict(epoch_metrics, loss, output, y, mask, t0, pre='val'):
    acc = calculate_acc(output, y, mask)
    (prec_macro, recall_macro, f_macro,
     prec_micro, recall_micro, f_micro,
     prec_weigh, recall_weigh, f_weigh) = \
        calculate_prec_recall_fscore(output, y, mask)
    # Update epoch metrics dict
    epoch_metrics[pre + '_loss'] += loss
    epoch_metrics[pre + '_acc'] += acc
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


def build_gnn(candidate_arch, search_space, in_feats, n_classes, in_drop):
    if search_space == 'macro':
        model = GraphNet(candidate_arch, in_feats, n_classes,
                         drop_out=in_drop, multi_label=False,
                         batch_normal=False, residual=False)
    elif search_space == 'micro':
        print('building micro model', candidate_arch)
        model_actions = candidate_arch['action']
        param = candidate_arch['hyper_param']
        layers = candidate_arch['layers']
        in_drop = param[1]
        num_hidden = param[3]
        model = MicroGNN(model_actions,
                         in_feats,
                         n_classes,
                         layers=layers,
                         num_hidden=num_hidden,
                         dropout=in_drop)
        print('finished building micro model')
    return model


def run_one_epoch(model, data, optimizer, epoch_metrics, dur, t0):
    # forward prop.
    logits = model(data.x, data.edge_index)
    # print('before softmax: ',
    #       logits[data.train_mask].shape,
    #       data.y[data.train_mask].shape)
    logits = F.log_softmax(logits, 1)
    # print('after softmax: ',
    #       logits[data.train_mask].shape,
    #       data.y[data.train_mask].shape)
    loss = loss_fn(logits[data.train_mask],
                   data.y[data.train_mask])
    optimizer.zero_grad()
    # Back prop.
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    epoch_metrics['train_loss'] += train_loss
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
        update_metrics_dict(epoch_metrics, val_loss, output,
                            data.y, data['valid_mask'], t0)
    del val_loss
    del output
    del logits
    torch.cuda.empty_cache()


def train(model, data, optimizer, epochs, n_batches=1, info=False):
    dur = []
    begin_time = time.time()
    min_val_loss = float("inf")
    model_val_acc = 0
    # Get batches
    print(data)
    if n_batches > 1:
        train_loader = split_and_batch_data(data, batches=n_batches)
    else:
        print('Loading full batch data to device...')
        data.to(device)
        print('Loading full batch data to device...DONE')
    model.train()
    loss_arr = np.zeros(epochs)
    for epoch in range(1, epochs + 1):
        epoch_metrics = defaultdict(float)
        t0 = time.time()
        if n_batches > 1:
            for data in train_loader:
                # Move batch to device
                data.to(device)
                run_one_epoch(model=model, data=data, optimizer=optimizer,
                              epoch_metrics=epoch_metrics, dur=dur, t0=t0)
            # Normalize epoch metrics by number of batches
            epoch_metrics = {k: (v / n_batches)
                             for k, v in epoch_metrics.items()}
        else:
            run_one_epoch(model=model, data=data, optimizer=optimizer,
                          epoch_metrics=epoch_metrics, dur=dur, t0=t0)
        epoch_metrics['train_time'] = time.time() - begin_time
        # Add current epoch's loss to loss array
        loss_arr[epoch - 1] = epoch_metrics['train_loss']
        if epoch_metrics['val_loss'] < min_val_loss:
            min_val_loss = epoch_metrics['val_loss']
            model_val_acc = epoch_metrics['val_acc']
        if info:
            print(epoch_info_str.format(epoch,
                                        epoch_metrics['val_loss'],
                                        np.mean(dur),
                                        epoch_metrics['train_acc'],
                                        epoch_metrics['acc'],
                                        0.0))
    # Add loss array to epoch metrics
    epoch_metrics['train_loss'] = list(loss_arr)
    end_time = time.time()
    print("Each Epoch Cost Time: %f " %
          ((end_time - begin_time) / epoch))
    print("val_score:", str(model_val_acc))
    return model, epoch_metrics


def test_with_data(model, data, test_metrics, t0):
    # Evaluate model on Test set (of the current BATCH, or full batch)
    with torch.no_grad():
        model.eval()
        logits = model(data.x, data.edge_index)
        logits = F.log_softmax(logits, 1)
        test_loss = loss_fn(logits[data['test_mask']],
                            data.y[data['test_mask']])
        test_loss = test_loss.item()
        # Calculate metrics on test dataset
        _, output = torch.max(logits, dim=1)
        update_metrics_dict(test_metrics, test_loss, output,
                            data.y, data['test_mask'], t0,
                            pre='test')
        del logits
        del output


def test(model, data, n_batches=1, show_info=False):
    # Get batches
    if n_batches > 1:
        test_loader = split_and_batch_data(data,
                                           batches=n_batches)
    test_metrics = defaultdict(float)
    t0 = time.time()
    if n_batches > 1:
        for data in test_loader:
            data.to(device)
            test_with_data(model=model, data=data,
                           test_metrics=test_metrics, t0=t0)
        test_metrics = {k: (v / n_batches)
                        for k, v in test_metrics.items()}
    else:
        test_with_data(model=model, data=data,
                       test_metrics=test_metrics, t0=t0)
    return test_metrics
