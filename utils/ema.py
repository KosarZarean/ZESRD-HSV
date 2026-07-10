
import torch


class EMA:

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()


    def update(self):

        for name, param in self.model.named_parameters():

            if param.requires_grad:

                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    +
                    (1.0 - self.decay) * param.data
                )


    def apply_shadow(self):

        for name, param in self.model.named_parameters():

            if param.requires_grad:
                param.data = self.shadow[name]


    def restore(self):

        for name, param in self.model.named_parameters():

            if param.requires_grad:
                param.data = self.shadow[name]
