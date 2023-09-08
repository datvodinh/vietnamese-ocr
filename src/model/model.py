import torch
import torch.nn as nn
import math

from src.backbone.resnet import ResNet18, ResNet50
from src.backbone.vgg import VGG19
from src.backbone.swin_transformer import SwinTransformer
from src.backbone.swin_transformer_v2 import SwinTransformerV2
from src.backbone.swin_encoder import SwinTransformerBackbone, SwinTransformerBackbone_v2

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

    def forward(self, x):
        seq_len = x.size(0)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(-1).expand(x.size()[:2])
        x = x + self.pos_embed(pos)
        return self.dropout(x)
    
class LayerNorm(nn.Module):
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
            self.cnn = ResNet18().to(device)
            self.cnn_conv = nn.Conv2d(512,config_trans['embed_size'],1).to(device)
        elif self.encoder_type == 'resnet50':
            self.cnn = ResNet50().to(device)
            self.cnn_conv = nn.Conv2d(2048,config_trans['embed_size'],1).to(device)
        elif self.encoder_type == 'vgg':
            self.cnn = VGG19().to(device)
            self.cnn_conv = nn.Conv2d(512,config_trans['embed_size'],1).to(device)
        elif self.encoder_type == 'swin_transformer':
            self.encoder = SwinTransformerBackbone(config_enc['swin']).to(device)
            embed_dim_swin = config_enc['swin']['embed_dim'] * 2 ** (len(config_enc['swin']['depths'])-1)
            embed_dim_trans = config_trans['embed_size']

            if embed_dim_swin > embed_dim_trans:
                self.swin_conv = True
                self.out_conv = nn.Conv1d(embed_dim_swin,embed_dim_trans,1).to(device)
            else:
                self.swin_conv = False
        elif self.encoder_type == 'swin_transformer_v2':
            self.encoder = SwinTransformerBackbone_v2(config_enc['swin']).to(device)
            embed_dim_swin = config_enc['swin']['embed_dim'] * 2 ** (len(config_enc['swin']['depths'])-1)
            embed_dim_trans = config_trans['embed_size']

            if embed_dim_swin > embed_dim_trans:
                self.swin_conv = True
                self.out_conv = nn.Conv1d(embed_dim_swin,embed_dim_trans,1).to(device)
            else:
                self.swin_conv = False
            
    def forward(self,x):

        if self.encoder_type in ['resnet18','resnet50','vgg']:
            out_cnn = self.cnn(x) # B C H/32 W/32
            out_cnn = self.cnn_conv(out_cnn).flatten(2)
            out_cnn = out_cnn.permute(0,2,1) # B L C
            x_embed = self.position_embed(out_cnn)
            out = self.encoder(x_embed) # B L C

        elif self.encoder_type in ['swin_transformer','swin_transformer_v2']:
            out = self.encoder(x).flatten(2) # B C H/32*W/32
            if self.swin_conv:
                out = self.out_conv(out).permute(0,2,1) # B H/32*W/32 C_out
            else:
                out = out.permute(0,2,1)

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
        x_embed = self.embed(x) * math.sqrt(self.d_model) # (B,L,E)
        out = self.position_embed(x_embed) # (B,L,E)
        out = self.decoder(out,encoder_out,tgt_mask = target_mask,tgt_key_padding_mask = padding) # (B,L,E)
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
    def _target_mask(self,target):
        mask = (torch.triu(torch.ones(target.shape[1], target.shape[1])) == 0).transpose(0, 1)
        return mask.bool().to(self.device)
    
    
    def forward(self,src,target,tar_pad=None,mode='train'):
        if mode == 'train':
            encoder_out = self.encoder(src) # (B,H/32 * W/32,C)
            logits = self._forward_decoder(target   = target,
                                        encoder_out = encoder_out,
                                        target_mask = self._target_mask(target),
                                        padding     = tar_pad,
                                        mode        = mode)
            return logits
        elif mode == 'predict':
            """
            src and target both in form batch dictionary: {"file_name1": img1,...}
            """
            dict_target = self._autoregressive_forward(src,target)
            return dict_target
        
    def _forward_decoder(self,target,encoder_out,target_mask=None,padding=None,mode='train'):
        out_transformer = self.decoder(target,encoder_out,target_mask=target_mask,padding=padding) # (B,L,E)
        if (mode=="train"):
            out_transformer = out_transformer.reshape(out_transformer.shape[0] * out_transformer.shape[1],out_transformer.shape[2])
        logits = self.fc(out_transformer) # (B*L,V) or (B,L,V)
        return logits

    def _autoregressive_forward(self,src,target):
        dict_target = {}
        c = 0
        with torch.no_grad():
            src_in = torch.stack(list(src.values()))
            encoder_out = self.encoder(src_in)
            dict_enc_out = {s_key:e_out for s_key,e_out in zip(list(src.keys()),encoder_out)}
            while c<32:
                lst_key = list(target.keys())
                for k in lst_key:
                    if target[k][-1] == 1:
                        dict_target[k] = target.pop(k)
                        dict_enc_out.pop(k)
                if len(dict_enc_out)==0:
                    break

                tensor_encoder_out = torch.stack(list(dict_enc_out.values()))
                tensor_target = torch.stack(list(target.values()))
                logits = self._forward_decoder(target   = tensor_target,
                                            encoder_out = tensor_encoder_out,
                                            target_mask = self._target_mask(tensor_target),
                                            mode        = "predict") # (B,L,V)
                logits = logits[:,-1,:] 
                target_next = torch.argmax(logits,dim=-1,keepdim=True)
                target = {k: torch.cat([target[k],t_next]) for k,t_next in zip(list(target.keys()),target_next)}
                c+=1
        return dict_target
    
