import torch


def color_constancy_loss(img):

    r = img[:,0].mean()

    g = img[:,1].mean()

    b = img[:,2].mean()

    loss = (
        (r-g)**2 +
        (r-b)**2 +
        (g-b)**2
    )

    return loss
