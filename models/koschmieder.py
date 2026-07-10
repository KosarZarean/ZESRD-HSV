import torch


class KoschmiederModel:

    @staticmethod
    def forward(R, G, A, t):

        A_global = torch.mean(
            A,
            dim=(2,3),
            keepdim=True
        )

        I = (
            G*t +
            (1-t)*A_global
        ) * R

        return torch.clamp(I,0,1)


    @staticmethod
    def inverse(G,A,t):

        A_global = torch.mean(
            A,
            dim=(2,3),
            keepdim=True
        )

        G_restore = (
            G -
            (1-t)*A_global
        )/(t+1e-6)

        return torch.clamp(G_restore,0,1)
