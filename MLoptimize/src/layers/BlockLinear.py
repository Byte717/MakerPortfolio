import torch
import torch.nn as nn

class BlockLinear(nn.Module):
    def __init__(self, inFeatures: int, outFeatures: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if inFeatures % 3:
            raise Exception("input has to be divisible by 3")
        if outFeatures % 3 or outFeatures != inFeatures:
            raise Exception("output has to be divisible by 3")
       
        self.numComponents:int = inFeatures//3
        self.components = nn.ModuleList(nn.Linear(3, 3) for _ in range(self.numComponents))
   
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        chunks = x.split(3, dim=1) # split by 3 at a time
        out = [layer(chunk) for layer, chunk in zip(self.components, chunks)]
        return torch.cat(out, dim=1)
