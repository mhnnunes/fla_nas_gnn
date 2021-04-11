
from graphnas_variants.micro.micro_gnn import MicroGNN
from graphnas_variants.macro.pyg_gnn_model_manager import GeoCitationManager

class MicroCitationManager(GeoCitationManager):

    def __init__(self, args):
        super(MicroCitationManager, self).__init__(args)

    def build_gnn(self, actions):
        model = MicroGNN(actions, self.in_feats, self.n_classes,
                         layers=self.args.layers_of_child_model,
                         num_hidden=self.args.num_hidden,
                         dropout=self.args.in_drop)
        return model

    def train(self, actions=None, format="micro"):
        self.current_action = actions
        print(actions)
        model_actions = actions['action']
        param = actions['hyper_param']
        self.args.lr = param[0]
        self.args.in_drop = param[1]
        self.args.weight_decay = param[2]
        self.args.num_hidden = param[3]
        return super(GeoCitationManager, self).train(model_actions,
                                                     format=format)

    def record_action_info(self, origin_action, metrics):
        return super(GeoCitationManager,
                     self).record_action_info(self.current_action,
                                              metrics)

    def evaluate(self, actions=None, format="micro"):
        print(actions)
        model_actions = actions['action']
        param = actions['hyper_param']
        self.args.lr = param[0]
        self.args.in_drop = param[1]
        self.args.weight_decay = param[2]
        self.args.num_hidden = param[3]
        return super(GeoCitationManager, self).evaluate(model_actions,
                                                        format=format)
