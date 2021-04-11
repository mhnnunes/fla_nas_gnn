
import os
import glob
import time
import torch
import numpy as np
import scipy.signal
import graphnas.utils.tensor_utils as utils

from graphnas.utils.model_utils import EarlyStop
from graphnas.utils.model_utils import TopAverage

from graphnas.selectors.model_selector import ModelSelector
from graphnas.graphnas_controller import SimpleNASController

logger = utils.get_logger()


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


history = []


def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


class RL_Selector(ModelSelector):
    """Manage the training process"""

    def __init__(self, args, search_space, action_list, submodel_manager):
        """
        Constructor for training algorithm.
        Build sub-model manager and controller.
        Build optimizer and cross entropy loss for controller.

        Args:
            args: From command line, picked up by `argparse`.
        """
        self.args = args
        self.controller_step = 0  # counter for controller
        self.cuda = args.cuda
        self.epoch = 0
        self.start_epoch = 0
        self.submodel_manager = None
        self.controller = None

        self.early_stop_manager = EarlyStop(10)
        self.reward_manager = TopAverage(10)

        super(RL_Selector, self).__init__(args,
                                          search_space,
                                          action_list,
                                          submodel_manager)
        self.build_model()  # build controller
        self.max_length = self.args.shared_rnn_max_length

        controller_optimizer = _get_optimizer(self.args.controller_optim)
        self.controller_optim = \
            controller_optimizer(self.controller.parameters(),
                                 lr=self.args.controller_lr)

    def build_model(self):
        # CALLS THIS ONE
        self.args.share_param = False
        self.args.shared_initial_step = 0
        self.controller = SimpleNASController(self.args,
                                              action_list=self.action_list,
                                              search_space=self.search_space,
                                              cuda=self.args.cuda)
        if self.cuda:
            self.controller.cuda()

    def train(self):
        """
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            start_epoch_time = time.time()
            # 1. Training the shared parameters of the child graphnas
            self.train_shared(max_step=self.args.shared_initial_step)
            # 2. Training the controller parameters theta
            self.train_controller()

            if self.epoch % self.args.save_epoch == 0:
                self.save_model()
            end_epoch_time = time.time()
            print("epoch ", str(self.epoch),
                  " took: ", str(end_epoch_time - start_epoch_time))

        self.save_model()

    def train_shared(self, max_step=50, gnn_list=None):
        """
        Args:
            max_step: Used to run extra training steps as a warm-up.
            gnn: If not None, is used instead of calling sample().

        """
        if max_step == 0:  # no train shared
            return

        print("*" * 35, "training model", "*" * 35)
        gnn_list = gnn_list if gnn_list else self.controller.sample(max_step)

        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)
            try:
                _, val_score = \
                    self.submodel_manager.train(gnn,
                                                format=self.args.format)
                logger.info(str(gnn) + ", val_score:" + str(val_score))
            except RuntimeError as e:
                if 'CUDA' in str(e):  # usually CUDA Out of Memory
                    print(e)
                else:
                    raise e

        print("*" * 35, "training over", "*" * 35)

    def get_reward(self, gnn_list, entropies, hidden):
        """
        Computes the reward of a single sampled model on validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
        if isinstance(gnn_list, dict):
            gnn_list = [gnn_list]
        if isinstance(gnn_list[0], list) or isinstance(gnn_list[0], dict):
            pass
        else:
            gnn_list = [gnn_list]  # when structure_list is one structure

        reward_list = []
        for gnn in gnn_list:
            gnn = self.form_gnn_info(gnn)
            val_acc, metrics = \
                self.submodel_manager.train(
                    gnn,
                    format=self.args.format)
            # Manage Hall of Fame
            if self.args.opt_metric not in metrics:
                print("Could not find optimization metric",
                      self.args.opt_metric, "in metrics dict.")
                reward = self.reward_manager.get_reward(0)
            else:
                self.hof.add(gnn, metrics[self.args.opt_metric])
                # Calculate reward in terms of the optimization metric selected
                reward = self.reward_manager.get_reward(
                    metrics[self.args.opt_metric])
            reward_list.append(reward)

        if self.args.entropy_mode == 'reward':
            rewards = reward_list + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(
                'Unkown entropy mode:' + str(self.args.entropy_mode))

        return rewards, hidden

    def train_controller(self):
        """
            Train controller to find better structure.
        """
        print("*" * 35, "training controller", "*" * 35)
        model = self.controller
        model.train()

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.controller.init_hidden(self.args.batch_size)
        total_loss = 0
        for step in range(self.args.controller_max_step):
            # sample graphnas
            structure_list, log_probs, entropies = \
                self.controller.sample(with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            results = self.get_reward(structure_list, np_entropies, hidden)
            torch.cuda.empty_cache()

            if results:  # has reward
                rewards, hidden = results
            else:
                # CUDA Error happens, drop structure
                # and step into next iteration
                continue

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            history.append(adv)
            adv = scale(adv, scale_value=0.5)
            adv_history.extend(adv)

            adv = utils.get_variable(adv, self.cuda, requires_grad=False)
            # policy loss
            loss = -log_probs * adv
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            self.controller_step += 1
            torch.cuda.empty_cache()

        print("*" * 35, "training controller over", "*" * 35)

    def evaluate(self, gnn):
        """
        Evaluate a structure on the validation set.
        """
        self.controller.eval()
        gnn = self.form_gnn_info(gnn)
        reward, scores, metrics = \
            self.submodel_manager.train(gnn,
                                        format=self.args.format)
        logger.info("".join(['eval | ', str(gnn),
                             ' | reward: {:8.2f}'.format(reward),
                             ' | scores: {:8.2f}'.format(scores)]))

    @property
    def controller_path(self):
        return "".join([str(self.args.dataset),
                        "/controller_epoch",
                        str(self.epoch),
                        "_step",
                        str(self.controller_step),
                        ".pth"])

    @property
    def controller_optimizer_path(self):
        return "".join([str(self.args.dataset),
                        "/controller_epoch",
                        str(self.epoch),
                        "_step",
                        str(self.controller_step),
                        "_optimizer.pth"])

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.dataset, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0])
                     for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):

        torch.save(self.controller.state_dict(), self.controller_path)
        torch.save(self.controller_optim.state_dict(),
                   self.controller_optimizer_path)

        logger.info('[*] SAVED: ' + str(self.controller_path))

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.dataset,
                             '*_epoch' + str(epoch) + '_*.pth'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            logger.info(
                '[!] No checkpoint found in ' + str(self.args.dataset) + '...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.controller_step = max(controller_steps)

        self.controller.load_state_dict(
            torch.load(self.controller_path))
        self.controller_optim.load_state_dict(
            torch.load(self.controller_optimizer_path))
        logger.info('[*] LOADED: ' + str(self.controller_path))
