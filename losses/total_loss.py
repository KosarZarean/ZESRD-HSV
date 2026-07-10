class TotalLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.vgg = VGGPerceptualLoss()

        self.edge = EdgeLoss()

    def forward(

        self,

        I,

        enhanced,

        R,

        G,

        R1,

        G1

    ):

        Lrec = physics_loss(I,R,G)

        Ltv = smoothness_loss(G,R)

        Lcons = consistency_loss(R,R1,G,G1)

        Lvgg = self.vgg(enhanced,I)

        Ledge = self.edge(enhanced,I)

        total = (

            6.0*Lrec+

            1.0*Ltv+

            1.5*Lcons+

            0.2*Lvgg+

            0.3*Ledge

        )

        return total
