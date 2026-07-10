import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):

    def __init__(self):

        super().__init__()

        sobel_x = torch.tensor(
            [[-1,0,1],
             [-2,0,2],
             [-1,0,1]],
            dtype=torch.float32
        )

        sobel_y = torch.tensor(
            [[-1,-2,-1],
             [0,0,0],
             [1,2,1]],
            dtype=torch.float32
        )

        self.register_buffer(
            "kx",
            sobel_x.view(1,1,3,3)
        )

        self.register_buffer(
            "ky",
            sobel_y.view(1,1,3,3)
        )

    def gradient(self,x):

        gx = F.conv2d(x,self.kx,padding=1)

        gy = F.conv2d(x,self.ky,padding=1)

        return torch.sqrt(gx**2+gy**2+1e-6)

    def forward(self,pred,target):

        return torch.mean(
            torch.abs(
                self.gradient(pred)-self.gradient(target)
            )
        )
