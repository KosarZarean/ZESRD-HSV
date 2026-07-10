def smoothness_loss(G,R):

    dx = torch.abs(G[:,:,:,1:]-G[:,:,:,:-1])

    dy = torch.abs(G[:,:,1:,:]-G[:,:,:-1,:])

    rx = torch.abs(R[:,:,:,1:]-R[:,:,:,:-1])

    ry = torch.abs(R[:,:,1:,:]-R[:,:,:-1,:])

    return (
        (dx*torch.exp(-10*rx)).mean()+
        (dy*torch.exp(-10*ry)).mean()
    )
