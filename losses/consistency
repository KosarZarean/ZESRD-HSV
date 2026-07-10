def consistency_loss(R,R1,G,G1):

    return (
        torch.mean(torch.abs(R-R1))+
        torch.mean(torch.abs(G-G1))
    )
