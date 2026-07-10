class TotalLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.vgg = VGGPerceptualLoss()
        self.edge = EdgeLoss()

    def forward(self, I, enhanced, R, G, R1, G1):

        # 1) Physics Reconstruction Loss
        Lrec = physics_loss(I, R, G)

        # 2) Smoothness Loss (Illumination + Reflectance)
        Ltv = smoothness_loss(G, R)

        # 3) Multi-stage Consistency Loss
        Lcons = consistency_loss(R, R1, G, G1)

        # 4) Perceptual Loss
        Lvgg = self.vgg(enhanced, I)

        # 5) Edge-aware Loss
        Ledge = self.edge(enhanced, I)

        total = (
            6.0 * Lrec +
            1.0 * Ltv +
            1.5 * Lcons +
            0.2 * Lvgg +
            0.3 * Ledge
        )

        return total
