import torch
import torch.nn as nn
class VisionTransformer(nn.Module):
    def __init__(self,patch_size,embed_size) -> None:
        super().__init__()
        
        self.embedding = nn.Conv2d(3,
                                embed_size,
                                kernel_size = patch_size,
                                stride = patch_size)
        
    def forward(self,img):
        emb = self.embedding(img)  # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c) # (b,l,c)
        return emb
