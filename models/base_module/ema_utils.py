

def copy_params(model_pairs):
    for model_pair in model_pairs:
        for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient


def _momentum_update(model_pairs, momentum=0.995):
    for model_pair in model_pairs:
        for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
            param_m.data = param_m.data * momentum + param.data * (1. - momentum)
