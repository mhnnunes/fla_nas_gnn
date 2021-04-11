"""Entry point."""


import torch
import argparse
import numpy as np
from sys import exit
from datetime import datetime
import graphnas.utils.tensor_utils as utils
from graphnas.utils.data_utils import available_datasets
from graphnas.utils.model_constants import optimization_metrics
# Selectors
from graphnas.selectors.rl_selector import RL_Selector
from graphnas.selectors.ea_selector import EA_Selector
from graphnas.selectors.gs_selector import GridSearch_Selector
from graphnas.selectors.rs_selector import RandomSearch_Selector
# Managers
from graphnas_variants.macro.pyg_gnn_model_manager import GeoCitationManager
# Search spaces
from graphnas_variants.macro.macro_search_space import MacroSearchSpace
from graphnas_variants.micro.micro_search_space import IncrementSearchSpace
from graphnas_variants.micro.micro_model_manager import MicroCitationManager


def build_args():
    parser = argparse.ArgumentParser(description='NAS for GNNs')
    register_default_args(parser)
    args = parser.parse_args()
    return args


def register_default_args(parser):
    parser.add_argument('--optimizer', type=str, default='EA',
                        choices=['EA', 'RL', 'RS', 'GS'],
                        help='EA: Evolutionary algorithm,\
                              RL: Reinforcement Learning algorithm\
                              RS: Random Search algorithm\
                              GS: Grid Search algorithm')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--max_save_num', type=int, default=5)
    # RS
    parser.add_argument('--load_from_file', type=str, default='',
                        help='Path of file containing RS architectures.')
    # EA
    parser.add_argument('--cycles', type=int, default=1000,
                        help='Evolution cycles')
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--sample_size', type=int, default=25,
                        help='Sample size for tournament selection')
    # Grid Search
    parser.add_argument('--n_processes', type=int, default=2)
    parser.add_argument('--process_index', type=int, default=0)
    # RL controller
    parser.add_argument('--hof_size', type=int, default=10)
    parser.add_argument('--save_epoch', type=int, default=2)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--format', type=str, default='two')
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--shared_initial_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_mode', type=str, default='reward',
                        choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)
    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--controller_max_step', type=int, default=100,
                        help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    # child model
    parser.add_argument('--layers_of_child_model', type=int, default=2)
    parser.add_argument('--opt_metric', type=str, default='val_acc',
                        choices=optimization_metrics,
                        help='Metric for selecting child models')
    parser.add_argument("--dataset", type=str, default="Citeseer",
                        choices=available_datasets,
                        required=False, help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--folds", type=int, default=1,
                        help="number of Cross Validation folds")
    parser.add_argument("--batches", type=int, default=1,
                        help="number of Mini-Batches for child models.")
    parser.add_argument("--retrain_epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # Admin
    parser.add_argument('--server_url', type=str, default="")
    parser.add_argument('--log_file', type=str,
                        default="log_file_" + datetime.now().strftime(
                            "%m_%d_%Y-%H:%M:%S") + ".txt")


def build_search_space(args):
    if args.search_mode == "macro":
        # generate model description in macro way
        # (generate entire network description)
        search_space_cls = MacroSearchSpace()
        search_space = search_space_cls.get_search_space()
        action_list = search_space_cls.generate_action_list(
            args.layers_of_child_model)
        # implements based on pyg
        submodel_manager = GeoCitationManager(args)
    if args.search_mode == "micro":
        args.format = "micro"
        args.predict_hyper = True
        if not hasattr(args, "num_of_cell"):
            args.num_of_cell = 2
        search_space_cls = IncrementSearchSpace()
        search_space = search_space_cls.get_search_space()
        submodel_manager = MicroCitationManager(args)
        search_space = search_space
        action_list = search_space_cls.generate_action_list(
            cell=args.num_of_cell)
        if hasattr(args, "predict_hyper") and args.predict_hyper:
            action_list = action_list + ["learning_rate",
                                         "dropout",
                                         "weight_decay",
                                         "hidden_unit"]
        else:
            action_list = action_list
    print("Search space:")
    print(search_space)
    print("Generated Action List: ")
    print(action_list)
    return search_space, action_list, submodel_manager


def main(args):  # pylint:disable=redefined-outer-name

    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    # Set the random seed for the program
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    np.set_printoptions(precision=8)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)
    search_space, action_list, submodel_manager = \
        build_search_space(args)

    if args.optimizer == 'EA':
        model_selector = EA_Selector(args,
                                     search_space,
                                     action_list,
                                     submodel_manager)
    elif args.optimizer == 'RL':
        model_selector = RL_Selector(args,
                                     search_space,
                                     action_list,
                                     submodel_manager)
    elif args.optimizer == 'GS':
        model_selector = GridSearch_Selector(args,
                                             search_space,
                                             action_list,
                                             submodel_manager)
    elif args.optimizer == 'RS':
        model_selector = RandomSearch_Selector(args,
                                               search_space,
                                               action_list,
                                               submodel_manager)
    else:
        raise Exception("[!] Optimizer not found: ", args.optimizer)
        exit(1)

    print('on selector main: ', args)
    model_selector.select()


if __name__ == "__main__":
    args = build_args()
    main(args)
