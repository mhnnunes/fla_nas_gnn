from graphnas.trainer import Trainer
from graphnas.graphnas_controller import SimpleNASController
from graphnas_variants.simple.simple_search_space import SimpleSearchSpace
from graphnas_variants.simple.simple_model_manager import SimpleCitationManager

class SimpleTrainer(Trainer):

    def build_model(self):

        if self.args.search_mode == "simple":
            self.submodel_manager = SimpleCitationManager(self.args)
            search_space_cls = SimpleSearchSpace()
            self.search_space = search_space_cls.get_search_space()
            self.action_list = search_space_cls.generate_action_list(
                self.args.layers_of_child_model)
            # build RNN controller
            self.controller = \
                SimpleNASController(self.args,
                                    action_list=self.action_list,
                                    search_space=self.search_space,
                                    cuda=self.args.cuda)
        if self.cuda:
            self.controller.cuda()

    def form_gnn_info(self, gnn):
        gnn_list = [gnn]
        state_length = len(self.search_space)
        result_gnn = []
        for gnn_info in gnn_list:
            predicted_gnn = {}
            gnn_layer_info = {}
            for index, each in enumerate(gnn_info):
                # current layer information is over
                if index % state_length == 0:
                    if gnn_layer_info:
                        predicted_gnn[index // state_length - 1] = \
                            gnn_layer_info
                        gnn_layer_info = {}
                gnn_layer_info[self.action_list[index]] = gnn_info[index]
            # add the last layer info
            predicted_gnn[index // state_length] = gnn_layer_info
            result_gnn.append(predicted_gnn)
        return result_gnn[0]

    @property
    def model_info_filename(self):
        return str(self.args.dataset) + "_" + \
            str(self.args.search_mode) + "_" + \
            str(self.args.format) + "_results.txt"
