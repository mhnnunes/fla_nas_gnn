
import torch

epoch_info_str = " | ".join(["Epoch {:05d}",
                             "Loss {:.4f}",
                             "Time(s) {:.4f}",
                             "acc {:.4f}",
                             "val_acc {:.4f}",
                             "test_acc {:.4f}"])

optimization_metrics = ['val_acc',         # Accuracy
                        'prec_macro',      # Macro avg. Precision
                        'prec_micro',      # Micro avg. Precision
                        'prec_weigh',      # Weighted avg. Precision
                        'recall_macro',    # Macro avg. Recall
                        'recall_micro',    # Micro avg. Recall
                        'recall_weigh',    # Weighted avg. Recall
                        'f_macro',         # Macro avg. F1
                        'f_micro',         # Micro avg. F1
                        'f_weigh',         # Weighted avg. F1
                        'avg_epoch_time',  # Weighted avg. F1
                        'train_time',      # Weighted avg. F1
                        ]


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")
