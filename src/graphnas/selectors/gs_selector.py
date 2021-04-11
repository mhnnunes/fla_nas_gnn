import time
import itertools
from math import ceil
from graphnas.selectors.model_selector import ModelSelector


class GridSearch_Selector(ModelSelector):
    """
    This class implements a Random Search method, on the Search Space
    provided to it.
    """
    def __init__(self, args, search_space, action_list, submodel_manager):
        super(GridSearch_Selector, self).__init__(args,
                                                  search_space,
                                                  action_list,
                                                  submodel_manager)
        self.n_processes = args.n_processes
        self.process_index = args.process_index

    def generate_individuals(self, action_list):
        for individual in itertools.product(*action_list):
            yield list(individual)

    def train_by_process(self, action_list, p_specific_list):
        start_time = time.time()
        best_ind_acc = 0.0
        best_ind = []
        all_individual_accs = []
        counter = 0
        print('action_list : ', action_list)
        for individual in enumerate(self.generate_individuals(action_list)):
            individual = self._generate_random_individual()
            ind_actions = self._construct_action([individual])
            gnn = self.form_gnn_info(ind_actions[0])
            ind_acc, metrics = \
                self.submodel_manager.train(gnn, format=self.args.format)
            all_individual_accs.append(ind_acc)
            if ind_acc > best_ind_acc:
                best_ind = individual.copy()
                best_ind_acc = ind_acc
        print('arch counter: ', counter)
        end_time = time.time()
        total_time = end_time - start_time
        p_specific_list.append(total_time)
        p_specific_list.append(best_ind)
        p_specific_list.append(best_ind_acc)
        p_specific_list.append(all_individual_accs)
        return

    def generate_process_list(self):
        action_list_indexes = \
            [list(range(len(self.search_space[action_list])))
             for action_list in self.action_list]
        print('\n\n Action List indexes!')
        print(action_list_indexes)
        div_first_per_process = \
            ceil(len(action_list_indexes[0]) / self.n_processes)
        process_list = []
        for index, l in enumerate(action_list_indexes):
            if index == 0:  # first action list on lists
                # take part of the first list equivalent to this process
                process_list.append(
                    l[(self.process_index * div_first_per_process):
                      ((self.process_index + 1) * div_first_per_process)])
            else:
                process_list.append(l)
        print('\n\np_lists: ')
        try:
            print(process_list, '\n')
        except Exception as e:
            print(e, repr(e))
        return process_list

    def train(self):
        print("\n\n===== Grid Search ====")
        print('starting processes')
        process_action_lists = self.generate_process_list()
        print(len(list(itertools.product(*process_action_lists))))
        # all_accuracies = []
        # for p_specific_list in process_return_lists:
        #     print('process name: ', p_specific_list[0])
        #     print('PID: ', p_specific_list[1])
        #     print('Elapsed time in this process: ' + str(p_specific_list[2]))
        #     print('[BEST STRUCTURE]', p_specific_list[3])
        #     print('[BEST STRUCTURE] Actions: ',
        #           self._construct_action([p_specific_list[3]]))
        #     print('[BEST STRUCTURE] Accuracy: ', p_specific_list[4])
        #     all_accuracies.extend(p_specific_list[5])
        # np.savetxt('all_accuracies.csv',
        #            np.array(all_accuracies),
        #            fmt='%.8f',
        #            delimiter=',',
        #            header='accuracy')
        print("===== Grid Search DONE ====")
