import time
import numpy as np
from collections import deque
from graphnas.utils.model_utils import construct_actions
from graphnas.selectors.model_selector import ModelSelector
from graphnas.utils.selector_utils import ScoredArchitecture


class EA_Selector(ModelSelector):
    """
    This class implements the Asyncronous Aging Evolution,
    proposed by Real et. al. on:

    Regularized Evolution for Image Classifier Architecture Search

    available on: https://arxiv.org/abs/1802.01548
    """
    def __init__(self, args, search_space, action_list, submodel_manager):
        super(EA_Selector, self).__init__(args,
                                          search_space,
                                          action_list,
                                          submodel_manager)
        self.random_seed = args.random_seed
        self.population = deque()
        self.population_size = args.population_size
        self.sample_size = args.sample_size
        self.cycles = args.cycles
        self.init_time = 0
        print('initializing population on evolution_trainer')
        self.__initialize_population()

    def _mutate_individual(self, indiv):
        # Choose a random position on the individual to mutate
        position_to_mutate = np.random.randint(len(indiv))
        # This position will receive a randomly chosen index
        # of the search_spaces's list
        # for the action corresponding to that position in the individual
        sp_list = self.search_space[self.action_list[position_to_mutate]]
        indiv[position_to_mutate] = \
            np.random.randint(0, len(sp_list))
        return indiv

    def _get_best_individual(self, sample):
        m = max(sample)
        return m.model, m.score

    def __initialize_population(self):
        print("\n\n===== Evaluating initial random population =====")
        start_initial_population_time = time.time()
        while len(self.population) < self.population_size:
            # print('adding individual #:', len(population))
            individual = self._generate_random_individual()
            ind_actions = construct_actions([individual],
                                            self.action_list,
                                            self.search_space)
            gnn = self.form_gnn_info(ind_actions[0])
            ind_acc, metrics = \
                self.submodel_manager.train(gnn, format=self.args.format)
            scored_arch = \
                ScoredArchitecture(
                    individual, metrics[self.args.opt_metric],
                    details="Initialization, {:d} individual".format(
                        len(self.population) + 1))
            self.population.append(scored_arch)
            print("individual:", gnn,
                  " val_score:", metrics[self.args.opt_metric])
            # Manage Hall of Fame
            self.hof.add_scored(scored_arch)
        end_initial_pop_time = time.time()
        self.init_time = end_initial_pop_time - start_initial_population_time
        print("Time elapsed initializing population: ",
              str(self.init_time))
        print("===== Evaluating initial random population DONE ====")

    def get_random_sample(self):
        sample = []  # list with population individuals
        indexes = np.random.randint(0,
                                    len(self.population),
                                    size=self.sample_size)
        for index in indexes:
            sample.append(self.population[index])
        return sample

    def train(self):
        print("\n\n===== Evolution ====")
        start_evolution_time = time.time()
        for cycle in range(self.cycles):
            # list with population individuals
            sample = self.get_random_sample()
            # Get best individual on sample to serve as parent
            parent, parent_score = \
                self._get_best_individual(sample)
            child = parent.copy()
            child = self._mutate_individual(child)
            gnn = \
                self.form_gnn_info(
                    construct_actions([child],
                                      self.action_list,
                                      self.search_space)[0])
            _, metrics = \
                self.submodel_manager.train(gnn, format=self.args.format)
            built_parent = \
                self.form_gnn_info(
                    construct_actions([parent],
                                      self.action_list,
                                      self.search_space)[0])
            print("parent: ", str(built_parent),
                  " val_score: ", str(parent_score), '|',
                  "child: ", str(gnn),
                  " val_score: ",
                  str(metrics[self.args.opt_metric]))
            # Add to Hall of Fame (if applicable)
            self.hof.add(
                gnn, metrics[self.args.opt_metric],
                details="Optimization, {:d} individual".format(cycle + 1))
            # Add to population (every child stays)
            self.population.append(
                ScoredArchitecture(
                    child, metrics[self.args.opt_metric],
                    details="Optimization, {:d} individual".format(cycle + 1)))
            # Remove oldest individual (Aging/Regularized evolution)
            self.population.popleft()
            scores = [a.score for a in self.population]
            print("[POPULATION STATS] Mean/Median/Best: ",
                  np.mean(scores),
                  np.median(scores),
                  np.max(scores))
        end_evolution_time = time.time()
        total_evolution_time = end_evolution_time - start_evolution_time
        print('Time spent on evolution: ',
              str(total_evolution_time))
        print('Total elapsed time: ',
              str(total_evolution_time + self.init_time))
        print("===== Evolution DONE ====")
