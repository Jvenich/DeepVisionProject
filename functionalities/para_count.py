

def count_para(model):
    """
    Counts all trainable parameters in a model.

    :param model: model of which the trainable parameters should be counted
    :return: number of trainable parameters
    """

    trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    n_trainable_parameters = sum([p.numel() for p in trainable_parameters])

    return n_trainable_parameters