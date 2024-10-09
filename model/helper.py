from model.loss import *
from model.model import *

def get_loss_fn(name):

    if name == "l2":
        return l2_loss
    elif name == "normed_l2":
        return normed_l2_loss
    elif name == "normed_semi_l2":
        return normed_semi_l2_loss
    elif name == "l1":
        return l1_loss
    elif name == "normed_l1":
        return normed_l1_loss
    elif name == "cross_l1":
        return cross_l1_loss
    elif name == "normed_cross_l1":
        return normed_cross_l1_loss
    elif name == "cross_l2":
        return cross_l2_loss
    elif name == "normed_cross_l2":
        return normed_cross_l2_loss
    else:
        raise ValueError(f"Unknown loss function: {name}")
    
def get_model(name, path, *args):

    if name == "DNR":
        model = DNR(*args)
    else:
        raise ValueError(f"Unknown model: {name}")

    model = model.cuda()
    if path != "":
        model.load_state_dict(torch.load(path))

    return model