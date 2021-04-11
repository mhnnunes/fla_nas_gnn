import time
from graphnas.utils.model_utils import process_action
from graphnas.utils.model_utils import construct_actions
from graphnas.selectors.model_selector import ModelSelector
from graphnas.utils.model_constants import optimization_metrics
# Server
from server_client.evaluator_client import EvaluatorClient


class RandomSearch_Selector(ModelSelector):
    """
    This class implements a Random Search method, on the Search Space
    provided to it.
    """
    def __init__(self, args, search_space, action_list, submodel_manager):
        super(RandomSearch_Selector, self).__init__(args,
                                                    search_space,
                                                    action_list,
                                                    submodel_manager)
        self.random_seed = args.random_seed
        self.cycles = args.cycles
        self.individuals = []
        self.server_proxy = \
            EvaluatorClient(self.args.server_url) \
            if self.args.server_url else None
        # Generate or load numerical individuals
        if not self.args.load_from_file:
            for _ in range(self.cycles):
                self.individuals.append(
                    self._generate_random_individual())
            self.save_individuals()
        else:
            self.load_individuals()
        # Transform to string version
        self.transform_individuals()

    def transform_individuals(self):
        # Calls construct_actions for the list of numerical individuals
        # which returns a list of string individuals.
        # Each individual is then passed to form_gnn_info.
        self.individuals = list(
            map(self.form_gnn_info,
                construct_actions(self.individuals,
                                  self.action_list,
                                  self.search_space)))

    def save_individuals(self):
        path = self.submodel_manager.record_action_info_filename
        path = path.replace('log_file', 'rs_individuals')
        with open(path, 'w') as dump:
            for ind in self.individuals:
                dump.write(str(ind))
                dump.write('\n')

    def load_individuals(self):
        with open(self.args.load_from_file, 'r') as f:
            for line in f.readlines():
                line_with_no_braces = line.replace('[', '').replace(']', '')
                self.individuals.append(
                    list(map(int, line_with_no_braces.split(','))))

    def train(self):
        print("\n\n===== Random Search ====")
        start_time = time.time()
        self.best_ind_score = 0.0
        self.best_ind = []
        for cycle, gnn in enumerate(self.individuals):
            if self.server_proxy is None:
                # evaluate locally if not using server
                # Ignore first return value to dump model
                _, metrics = \
                    self.submodel_manager.train(gnn, format=self.args.format)
            else:
                # Fix candidate arch before passing to server
                candidate_arch = process_action(
                    gnn, type='two', args=self.args)
                # Call server evaluation
                lr, in_drop, weight_decay = \
                    self.get_model_params(candidate_arch)
                results = self.server_proxy.evaluate_pipeline(
                    self.server_proxy.build_pipeline(
                        candidate_arch,
                        self.args.search_mode,
                        self.args.dataset,
                        self.args.random_seed,
                        self.args.folds,
                        self.args.batches,
                        lr,
                        in_drop,
                        weight_decay,
                        self.args.layers_of_child_model))
                print('server results for gnn ', gnn, results)
                metrics = {metric: 0.0 for metric in optimization_metrics}
                if not results['error']:
                    # If the model executed successfully,
                    # the error results will be empty
                    metrics = results['train']
                # Save to log
                self.submodel_manager.record_action_info(gnn, metrics)
            # Manage Hall of Fame
            self.hof.add(gnn, metrics[self.args.opt_metric],
                         details="Optimization, {:d} individual".format(
                         cycle + 1))
            print("individual:", gnn,
                  " val_score:", metrics[self.args.opt_metric])
            # Keep best individual in variable
            if metrics[self.args.opt_metric] > self.best_ind_score:
                self.best_ind = gnn.copy()
                self.best_ind_score = metrics[self.args.opt_metric]
        end_time = time.time()
        total_time = end_time - start_time
        print('Total elapsed time: ' + str(total_time))
        print('[BEST STRUCTURE]', self.best_ind)
        print('[BEST STRUCTURE] Accuracy: ', self.best_ind_score)
        print("===== Random Search DONE ====")
