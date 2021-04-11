

class MacroSearchSpace(object):
    def __init__(self, search_space=None):
        if search_space:
            self.search_space = search_space
        else:
            # Define operators in search space
            self.search_space = {
                "attention_type": ["gat",
                                   "gcn",
                                   "cos",
                                   "const",
                                   "gat_sym",
                                   'linear',
                                   'generalized_linear'],
                'aggregator_type': ["sum", "mean", "max", "mlp", ],
                'activate_function': ["sigmoid", "tanh", "relu", "linear",
                                      "softplus", "leaky_relu",
                                      "relu6", "elu"],
                'number_of_heads': [1, 2, 4, 6, 8, 16],
                'hidden_units': [4, 8, 16, 32, 64, 128, 256],
            }

    def get_search_space(self):
        return self.search_space

    # Assign operator category for controller RNN outputs.
    # The controller RNN will select operators from search
    # space according to operator category.
    def generate_action_list(self, num_of_layers=2):
        action_names = list(self.search_space.keys())
        action_list = action_names * num_of_layers
        return action_list
