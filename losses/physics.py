import torch


def physics_loss(I,R,G):

    reconstruction = R*G

    loss = torch.mean(
        torch.abs(
            reconstruction-I
        )
    )

    return loss
