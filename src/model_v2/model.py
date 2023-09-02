import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from src.backbone.resnet import resnet18,resnet50
from src.backbone.vgg import vgg19
from src.backbone.swin_transformer import SwinTransformer
from src.backbone.ViT import VisionTransformer
class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,device,dropout = 0.2,max_len=1000):
        super().__init__()
        PE = torch.zeros((1,max_len,num_hiddens)).to(device)
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0,max_len,dtype=torch.float32).reshape(-1,1) \
        / torch.pow(10000,torch.arange(0,num_hiddens,2,dtype=torch.float32) / num_hiddens)
        PE[:,:,0::2] = torch.sin(position)
        PE[:,:,1::2] = torch.cos(position)
        self.register_buffer('PE',PE)


    def forward(self,x):
        x = x + self.PE[:,:x.shape[1],:]
        return self.dropout(x)

class LearnableEmbedding(nn.Module):
    def __init__(self, d_model, device, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout).to(device)

        self.pos_embed = nn.Embedding(max_len, d_model).to(device)
        self.layernorm = LayerNorm(d_model).to(device)

    def forward(self, x):
        seq_len = x.size(0)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(-1).expand(x.size()[:2])
        x = x + self.pos_embed(pos)
        return self.dropout(self.layernorm(x))

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, d_model, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta  

class Encoder(nn.Module):
    def __init__(self,config_trans,config_enc,device):
        super().__init__()
        if config_trans['embed_type'] == 'position':
            self.position_embed = PositionalEncoding(num_hiddens = config_trans['embed_size'],
                                                     device      = device,
                                                     max_len     = config_trans['max_len'],
                                                     dropout     = config_trans['dropout'])
        elif config_trans['embed_type'] == "learned":
            self.position_embed = LearnableEmbedding(d_model = config_trans['embed_size'],
                                                     device  = device,
                                                     max_len = config_trans['max_len'],
                                                     dropout = config_trans['dropout'])
            
        encoder_layer = nn.TransformerEncoderLayer(d_model        = config_trans['embed_size'],
                                                  nhead           = config_trans['num_heads'],
                                                  dim_feedforward = config_trans['embed_size'] * 4,
                                                  device          = device,
                                                  dropout         = config_trans['dropout'],
                                                  batch_first     = True)


        self.encoder = nn.TransformerEncoder(encoder_layer = encoder_layer,
                                             num_layers    = config_trans['num_layers'])

        self.dropout = nn.Dropout(config_trans['dropout'])
        self.encoder_type = config_enc['type']
        if self.encoder_type == 'resnet18':
            self.cnn = resnet18().to(device)
            self.cnn_conv = nn.Conv2d(512,config_trans['embed_size'],1).to(device)
        elif self.encoder_type == 'resnet50':
            self.cnn = resnet50().to(device)
            self.cnn_conv = nn.Conv2d(2048,config_trans['embed_size'],1).to(device)
        elif self.encoder_type == 'vgg':
            self.cnn = vgg19().to(device)
            self.cnn_conv = nn.Conv2d(512,config_trans['embed_size'],1).to(device)
        elif self.encoder_type == 'swin_transformer':
            self.encoder = SwinTransformer(img_size    = config_enc['swin']['img_size'],
                                           embed_dim   = config_enc['swin']['embed_dim'],
                                           window_size = config_enc['swin']['window_size'],
                                           in_chans    = config_enc['swin']['in_channels'],
                                           drop_rate   = config_enc['swin']['dropout']).to(device)
        elif self.encoder_type == 'vision_transformer':
            self.vit = VisionTransformer(patch_size = 16,
                                         embed_size = config_trans['embed_size']).to(device)
        
    def forward(self,x,mask=None,padding=None):
        '''
        Perform a forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor to the encoder.
            mask (torch.Tensor, optional): Mask for the input sequence.
            padding (torch.Tensor, optional): Padding mask for the input sequence.
        
        Returns:
            out (torch.Tensor): Output tensor from the encoder.
        '''
        if self.encoder_type in ['resnet18','resnet50','vgg']:
            out_cnn = self.cnn(x)
            out_cnn = self.cnn_conv(out_cnn).flatten(2)
            out_cnn = out_cnn.permute(0,2,1)
            x_embed = self.position_embed(out_cnn)
            out = self.dropout(x_embed)
            out = self.encoder(out).permute(1,0,2)

        elif self.encoder_type == 'swin_transformer':
            out = self.encoder(x)
        elif self.encoder_type == 'vision_transformer':
            out_vit = self.vit(x)
            x_embed = self.position_embed(out_vit)
            out = self.dropout(x_embed)
            out = self.encoder(out).permute(1,0,2)

        return out
    
class Decoder(nn.Module):
    def __init__(self,config_trans,vocab_size,device):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,config_trans['embed_size']).to(device)
        if config_trans['embed_type'] == 'position':
            self.position_embed = PositionalEncoding(config_trans['embed_size'],device,max_len=config_trans['max_len'],dropout=config_trans['dropout'])
        elif config_trans['embed_type'] == "learned":
            self.position_embed = LearnableEmbedding(config_trans['embed_size'],device,max_len=config_trans['max_len'],dropout=config_trans['dropout'])
        decoder_layer = nn.TransformerDecoderLayer(d_model        = config_trans['embed_size'],
                                                  nhead           = config_trans['num_heads'],
                                                  dim_feedforward = config_trans['embed_size'] * 4,
                                                  device          = device,
                                                  dropout         = config_trans['dropout'],
                                                  batch_first     = True
                                                  )

        self.decoder = nn.TransformerDecoder(decoder_layer = decoder_layer,
                                             num_layers    = config_trans['num_layers'])
        self.d_model = config_trans['embed_size']

    def forward(self,x,encoder_out,target_mask=None,padding=None):
        '''
        Perform a forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Input tensor to the decoder.
            encoder_out (torch.Tensor): Output tensor from the encoder.
            src_mask (torch.Tensor, optional): Mask for the source sequence.
            target_mask (torch.Tensor, optional): Mask for the target sequence.
            padding (torch.Tensor, optional): Padding mask.
        
        Returns:
            out (torch.Tensor): Output tensor from the decoder.
        '''
        x_embed = self.embed(x) * math.sqrt(self.d_model)
        out = self.position_embed(x_embed)
        out = self.decoder(out,encoder_out,tgt_mask = target_mask,tgt_key_padding_mask = padding)
        
        return out
    
class OCRTransformerModel(nn.Module):
    def __init__(self,config,vocab_size):
        super().__init__()
        self.encoder = Encoder(config["transformer"],config["encoder"],config["device"])
        self.decoder = Decoder(config["transformer"],vocab_size,config["device"])
        self.fc = nn.Linear(config["transformer"]['embed_size'],vocab_size).to(config["device"])
        self.apply(self._init_weights)
        self.device = config["device"]
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    def target_mask(self,target):
        mask = (torch.triu(torch.ones(target.shape[1], target.shape[1])) == 0).transpose(0, 1)
        return mask.bool().to(self.device)
    
    
    def forward(self,src,target,tar_pad=None,mode='train'):
        if mode == 'train':
            encoder_out = self.encoder(src)
            out_transformer = self.decoder(target,encoder_out,target_mask=self.target_mask(target),padding=(tar_pad==0) if tar_pad is not None else None)
            out_transformer = out_transformer.reshape(out_transformer.shape[0] * out_transformer.shape[1],out_transformer.shape[2])
            out = self.fc(out_transformer)
            return out
        elif mode == 'predict':
            with torch.no_grad():
                encoder_out = self.encoder(src)
                c = 0
                while target[0][-1] != 1 and c < 30: # <eos>
                    out_transformer = self.decoder(target,encoder_out,target_mask=self.target_mask(target))
                    out_transformer = out_transformer.reshape(out_transformer.shape[0] * out_transformer.shape[1],out_transformer.shape[2])
                    logits = self.fc(out_transformer)
                    logits = logits[-1,:]
                    target_next = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
                    target = torch.cat((target, target_next), dim=1).to(self.device)
                    c+=1
            return target[0]
    
