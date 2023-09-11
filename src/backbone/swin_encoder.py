import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer,SwinTransformerBlockV2,SwinTransformerBlock,PatchMerging,PatchMergingV2
class SwinTransformerBackbone(nn.Module):
    def __init__(self,config_swin):
        super().__init__()
        self.model = SwinTransformer(patch_size=[4,4],
                                    embed_dim=config_swin['embed_dim'],
                                    depths=config_swin['depths'],
                                    num_heads=config_swin['num_heads'],
                                    window_size=config_swin['window_size'],
                                    dropout=config_swin['dropout'],
                                    block=SwinTransformerBlock,
                                    downsample_layer=PatchMerging)

    def forward(self,x):
        x = self.model.features(x) # B H W C
        x = self.model.permute(x)  # B C H W
        return x
    
class SwinTransformerBackbone_v2(nn.Module):
    def __init__(self,config_swin):
        super().__init__()
        self.model = SwinTransformer(patch_size=[4,4],
                                    embed_dim=config_swin['embed_dim'],
                                    depths=config_swin['depths'],
                                    num_heads=config_swin['num_heads'],
                                    window_size=config_swin['window_size'],
                                    dropout=config_swin['dropout'],
                                    block=SwinTransformerBlockV2,
                                    downsample_layer=PatchMergingV2)

    def forward(self,x):
        x = self.model.features(x) # B H W C
        x = self.model.permute(x)  # B C H W
        return x