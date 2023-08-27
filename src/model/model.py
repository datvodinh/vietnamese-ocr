import torch
import torch.nn as nn
from src.backbone.resnet import resnet18,resnet50
from src.backbone.vgg import vgg19
from src.backbone.swin_transformer import SwinTransformer
from src.backbone.ViT import VisionTransformer
class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,device,dropout = 0.5,max_len=1000):
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
        self.dropout = nn.Dropout(p=dropout)

        self.pos_embed = nn.Embedding(max_len, d_model).to(device)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(0)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(-1).expand(x.size()[:2])
        x = x + self.pos_embed(pos)
        return self.dropout(self.layernorm(x))

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_size,heads,device,bias=False):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = int(embed_size / heads)
        self.keys = nn.Linear(embed_size,embed_size,bias=bias).to(device)
        self.queries = nn.Linear(embed_size,embed_size,bias=bias).to(device)
        self.values = nn.Linear(embed_size,embed_size,bias=bias).to(device)
        self.fc = nn.Linear(embed_size,embed_size,bias=bias).to(device)

    def forward(self,query,key,value,mask=None,padding=None):
        '''
        - Overview: Calculate multi-head attention.
            
        - Arguments:
            - query: (`torch.Tensor`): `(batch_size,query_len,embed_size)`
            - key: (`torch.Tensor`): `(batch_size,key_len,embed_size)`
            - value: (`torch.Tensor`): `(batch_size,value_len,embed_size)`
        - Return:
            - out (attention score) `torch.Tensor`: `(batch_size,query_len,embed_size)`

        '''

        keys = self.keys(key).reshape(key.shape[0],key.shape[1],self.heads,self.heads_dim) # (batch_size,key_len,heads,head_dim)
        queries = self.queries(query).reshape(query.shape[0],query.shape[1],self.heads,self.heads_dim) # (batch_size,query_len,heads,head_dim)
        values = self.values(value).reshape(value.shape[0],value.shape[1],self.heads,self.heads_dim) # (batch_size,value_len,heads,head_dim)

        keys = keys / (self.embed_size)**(1/4)
        queries = queries / (self.embed_size)**(1/4)
        dot_product = torch.einsum('bkhd,bqhd->bhqk',keys,queries) # (batch_size,heads,query_len,key_len)
        if mask is not None:
            dot_product = dot_product.masked_fill(mask==0,float('-1e20'))
        if padding is not None:
            dot_product = dot_product.masked_fill(padding==0,float('-1e20'))
        scaled_product = torch.softmax(dot_product ,dim=-1)
        alpha = torch.einsum("bhqk,bvhd->bqhd",scaled_product,values)
        out = self.fc(alpha.reshape(query.shape[0],query.shape[1],self.embed_size))
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, bias,device):
        """
        Overview: Initialize a Transformer Block.
        
        Arguments:
            embed_dim (`int`): Dimensionality of the input embeddings.
            num_heads (`int`): Number of attention heads.
        """
        super().__init__()
        self.attention   = MultiHeadAttention(embed_dim, num_heads,device, bias)
        self.layer_norm1 = nn.LayerNorm(embed_dim).to(device)
        self.layer_norm2 = nn.LayerNorm(embed_dim).to(device)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim,embed_dim)
        ).to(device)

    def forward(self, query, key, value, mask=None,padding=None):
        """
        - Overview: Forward pass of the Transformer Block.
        
        - Arguments:
            - query: (`torch.Tensor`): `(batch_size,query_len,embed_size)`
            - key: (`torch.Tensor`): `(batch_size,key_len,embed_size)`
            - value: (`torch.Tensor`): `(batch_size,value_len,embed_size)`
        
        - Returns:
            out (tensor): Output tensor after the Transformer Block.
        """
        a_score  = self.attention(query,key,value, mask,padding)
        out      = self.layer_norm1(a_score + query)
        out_ffn  = self.fc(out)
        out      = self.layer_norm2(out + out_ffn)
        assert torch.isnan(out).any() == False, "Transformer block returned NaN!"

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,dropout,device,bias=False):
        super().__init__()
        self.transformer_block = TransformerBlock(embed_size,heads,bias,device)
        self.attention = MultiHeadAttention(embed_size,heads,device,bias)
        self.layer_norm = nn.LayerNorm(embed_size).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,enc_value,enc_key,src_mask=None,target_mask=None,padding=None):
        '''
        Perform a forward pass through the decoder block.
        
        Args:
            x (torch.Tensor): Input tensor to the decoder block.
            enc_value (torch.Tensor): Encoded values from the encoder.
            enc_key (torch.Tensor): Encoded keys from the encoder.
            src_mask (torch.Tensor, optional): Mask for source sequence.
            target_mask (torch.Tensor, optional): Mask for target sequence.
            padding (torch.Tensor, optional): Padding mask.
        
        Returns:
            out (torch.Tensor): Output tensor from the decoder block.
        '''
        out = self.layer_norm(x + self.attention(x,x,x,src_mask,padding))
        out = self.dropout(out)
        out = self.transformer_block(query   = out,
                                     key     = enc_key,
                                     value   = enc_value)

        return out

class Encoder(nn.Module):
    def __init__(self,embed_size,heads,num_layers,max_len,dropout,device,bias=False,embed_type='position'):
        super().__init__()
        if embed_type == 'position':
            self.position_embed = PositionalEncoding(embed_size,device,max_len=max_len,dropout=dropout)
        elif embed_type == "learned":
            self.position_embed = LearnableEmbedding(embed_size,device,max_len=max_len,dropout=dropout)
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(embed_size,heads,bias,device)
                for _ in range(num_layers)
            ]

        )

        self.dropout = nn.Dropout(dropout)

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
        x_embed = self.position_embed(x)
        out = self.dropout(x_embed)
        for layer in self.encoder_layers:
            out = layer(out,out,out,mask,padding)
    
        return out
    
class Decoder(nn.Module):
    def __init__(self,vocab_size,embed_size,heads,num_layers,max_len,dropout,device,bias=False,embed_type='position'):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,embed_size).to(device)
        if embed_type == 'position':
            self.position_embed = PositionalEncoding(embed_size,device,max_len=max_len,dropout=dropout)
        elif embed_type == "learned":
            self.position_embed = LearnableEmbedding(embed_size,device,max_len=max_len,dropout=dropout)
        self.decoder_layer = nn.ModuleList(
            [
                DecoderBlock(embed_size,heads,dropout,device,bias)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,x,encoder_out,src_mask=None,target_mask=None,padding=None):
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
        x_embed = self.embed(x)
        x_embed = self.position_embed(x_embed)
        out = self.dropout(x_embed)
        for layer in self.decoder_layer:
            out = layer(out,encoder_out,encoder_out,src_mask,target_mask,padding)

        # out = self.fc(out)

        return out
    
class TransformerModel(nn.Module):
    def __init__(self,vocab_size,embed_size,heads,num_layers,max_len,dropout,bias,device,lr,batch_size,embed_type):
        
        super().__init__()
        self.encoder = Encoder(embed_size,heads,num_layers,max_len,dropout,device,bias,embed_type).to(device)
        self.decoder = Decoder(vocab_size,embed_size,heads,num_layers,max_len,dropout,device,bias,embed_type).to(device)
        self.apply(self._init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.device = device
        self.batch_size = batch_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def target_mask(self,target):
        batch_size,target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len,target_len))).expand(batch_size,1,target_len,target_len)
        return target_mask.to(self.device)
    
    def padding_mask(self,padding=None):
        if padding is not None:
            return (padding.transpose(2,1) @ padding).unsqueeze(1)
        else:
            return None
    
    def forward(self,src,target,tar_pad=None):
        encoder_out = self.encoder(src)
        out = self.decoder(target,encoder_out,target_mask=self.target_mask(target),padding=self.padding_mask(tar_pad))
        return out
    
class OCRModel(nn.Module):
    def __init__(self,config,vocab_size):
        super().__init__()
        self.transformer = TransformerModel(
            vocab_size  = vocab_size,
            embed_size  = config["transformer"]['embed_size'],
            heads       = config["transformer"]['num_heads'],
            num_layers  = config["transformer"]['num_layers'],
            max_len     = config["transformer"]['max_len'],
            dropout     = config["transformer"]['dropout'],
            device      = config['device'],
            lr          = config['lr'],
            batch_size  = config['batch_size'],
            bias        = config["transformer"]['bias'],
            embed_type  = config["transformer"]['embed_type']
        )

        self.model_type = config['encoder']['type']

        if self.model_type == 'resnet18':
            self.cnn = resnet18().to(config['device'])
            self.cnn_fc = nn.Linear(2048,config["transformer"]['embed_size']).to(config['device'])
        elif self.model_type == 'resnet50':
            self.cnn = resnet50().to(config['device'])
            self.cnn_fc = nn.Linear(2048,config["transformer"]['embed_size']).to(config['device'])
        elif self.model_type == 'vgg':
            self.cnn = vgg19().to(config['device'])
            self.cnn_fc = nn.Linear(512,config["transformer"]['embed_size']).to(config['device'])
        elif self.model_type == 'swin_transformer':
            self.encoder = SwinTransformer(img_size=(64,128),embed_dim=48,window_size=8).to(config['device'])
        elif self.model_type == 'vision_transformer':
            self.vit = VisionTransformer(config['ViT']['patch_size'],config["transformer"]['embed_size']).to(config['device'])
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config["transformer"]['embed_size'],vocab_size)
        ).to(config['device'])

    def forward(self,src,target,padding=None):

        if self.model_type in ['resnet18','resnet50','vgg']:
            out_cnn = self.cnn(src)
            out_cnn = self.cnn_fc(out_cnn)
            if padding is not None:
                out_transformer = self.transformer(out_cnn,target,padding.unsqueeze(1))
            else:
                out_transformer = self.transformer(out_cnn,target)

        elif self.model_type == 'swin_transformer':
            encoder_out = self.encoder(src)
            if padding is not None:
                padding = padding.unsqueeze(1)
            out_transformer = self.transformer.decoder(target,encoder_out,target_mask=self.transformer.target_mask(target),padding=self.transformer.padding_mask(padding))
        
        elif self.model_type == 'vision_transformer':
            out_vit = self.vit(src)
            if padding is not None:
                out_transformer = self.transformer(out_vit,target,padding.unsqueeze(1))
            else:
                out_transformer = self.transformer(out_vit,target)

        out_transformer = out_transformer.reshape(out_transformer.shape[0] * out_transformer.shape[1],out_transformer.shape[2])
        out = self.fc(out_transformer)
        return out
    


