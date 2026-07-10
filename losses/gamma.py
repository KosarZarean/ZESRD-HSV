import torch


def gamma_regularization(gamma):

    target = 0.50

    return torch.mean(
        (gamma - target) ** 2
    )
