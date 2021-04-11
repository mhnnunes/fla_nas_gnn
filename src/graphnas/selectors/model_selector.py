
import pickle
import numpy as np
from graphnas.utils.selector_utils import HallOfFame


class ModelSelector(object):
    """Manage the GNN child model selection process"""

    def __init__(self, args, search_space, action_list, submodel_manager):
        """
        Constructor for child model selection algorithm.
        Build sub-model manager and instantiates the search space.
        ATTENTION: The constructor of the subclasses
        are called BEFORE this one.
        Args:
            args: From command line, picked up by `argparse`.
        """
        self.args = args
        self.search_space = search_space
        self.action_list = action_list
        self.submodel_manager = submodel_manager
        self.hof = HallOfFame(self.args.hof_size)

    def get_model_params(self, candidate_arch):
        lr = self.args.lr \
            if self.args.search_mode == 'macro' \
            else candidate_arch['hyper_param'][0]
        in_drop = self.args.in_drop \
            if self.args.search_mode == 'macro' \
            else candidate_arch['hyper_param'][1]
        weight_decay = self.args.weight_decay \
            if self.args.search_mode == 'macro' \
            else candidate_arch['hyper_param'][2]
        return lr, in_drop, weight_decay

    def build_model(self):
        pass

    def _generate_random_individual(self):
        ind = []
        for action in self.action_list:
            ind.append(np.random.randint(0,
                                         len(self.search_space[action])))
        return ind

    def form_gnn_info(self, gnn):
        if self.args.search_mode == "micro":
            actual_action = {}
            if self.args.predict_hyper:
                actual_action["action"] = gnn[:-4]
                actual_action["hyper_param"] = gnn[-4:]
            else:
                actual_action["action"] = gnn
                actual_action["hyper_param"] = [0.005, 0.8, 5e-5, 128]
            return actual_action
        return gnn

    def dump_hall_of_fame(self):
        path = self.submodel_manager.record_action_info_filename
        path = path.replace('log_file', 'hall_of_fame')
        path = path.replace('.txt', '.pkl')
        print('Dumping hall of fame to pickle file...')
        print('Filename: ', path)
        with open(path, 'wb') as f:
            pickle.dump(self.hof, f)
        print('Dumping hall of fame to pickle file...DONE')

    def select(self):
        self.train()
        self.dump_hall_of_fame()
        print("\n\n====Printing HallOfFame====")
        print(self.hof.get_elements())
        print("\n\n")

    def train(self):
        pass

    def train_shared(self, max_step=50, gnn_list=None):
        pass

    def evaluate(self, gnn):
        pass
