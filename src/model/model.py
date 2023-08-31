import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.layernorm = nn.LayerNorm(d_model).to(device)

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
            dot_product = dot_product.masked_fill(padding[:,None,:,None]==0,float('-1e20'))
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
                                     value   = enc_value,
                                     padding = padding)

        return out

class Encoder(nn.Module):
    def __init__(self,config_trans,config_enc,device):
        super().__init__()
        if config_trans['embed_type'] == 'position':
            self.position_embed = PositionalEncoding(config_trans['embed_size'],device,max_len=config_trans['max_len'],dropout=config_trans['dropout'])
        elif config_trans['embed_type'] == "learned":
            self.position_embed = LearnableEmbedding(config_trans['embed_size'],device,max_len=config_trans['max_len'],dropout=config_trans['dropout'])
        self.encoder_layers = nn.ModuleList(
            [
                TransformerBlock(config_trans['embed_size'],config_trans['num_heads'],config_trans['bias'],device)
                for _ in range(config_trans['num_layers'])
            ]

        )

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
                                           in_chans=config_enc['swin']['in_channels']).to(device)
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
            for layer in self.encoder_layers:
                out = layer(out,out,out,mask,padding)

        elif self.encoder_type == 'swin_transformer':
            out = self.encoder(x)
        elif self.encoder_type == 'vision_transformer':
            out_vit = self.vit(x)
            x_embed = self.position_embed(out_vit)
            out = self.dropout(x_embed)
            for layer in self.encoder_layers:
                out = layer(out,out,out,mask,padding)

        return out
    
class Decoder(nn.Module):
    def __init__(self,config_trans,vocab_size,device):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,config_trans['embed_size']).to(device)
        if config_trans['embed_type'] == 'position':
            self.position_embed = PositionalEncoding(config_trans['embed_size'],device,max_len=config_trans['max_len'],dropout=config_trans['dropout'])
        elif config_trans['embed_type'] == "learned":
            self.position_embed = LearnableEmbedding(config_trans['embed_size'],device,max_len=config_trans['max_len'],dropout=config_trans['dropout'])
        self.decoder_layer = nn.ModuleList(
            [
                DecoderBlock(config_trans['embed_size'],config_trans['num_heads'],config_trans['dropout'],device,config_trans['bias'])
                for _ in range(config_trans['num_layers'])
            ]
        )

        self.dropout = nn.Dropout(config_trans['dropout'])

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
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def target_mask(self,target):
        batch_size,target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len,target_len))).expand(batch_size,1,target_len,target_len)
        return target_mask.to(self.device)
    
    
    def forward(self,src,target,tar_pad=None,mode='train',**kwargs):
        if mode == 'train':
            encoder_out = self.encoder(src)
            out_transformer = self.decoder(target,encoder_out,target_mask=self.target_mask(target),padding=tar_pad)
            out_transformer = out_transformer.reshape(out_transformer.shape[0] * out_transformer.shape[1],out_transformer.shape[2])
            out = self.fc(out_transformer)
            return out
        elif mode == 'predict':
            with torch.no_grad():
                encoder_out = self.encoder(src)
                c = 0
                while target[0][-1] != 1 and c < 30: # <eos>
                    out_transformer = self.decoder(target,encoder_out)
                    out_transformer = out_transformer.reshape(out_transformer.shape[0] * out_transformer.shape[1],out_transformer.shape[2])
                    logits = self.fc(out_transformer)
                    logits = logits[-1,:]
                    if kwargs['sampling'] == 'soft':
                        probs = F.softmax(logits / kwargs['temperature'], dim=-1)
                        target_next = torch.multinomial(probs, num_samples=1).unsqueeze(0) # (B, 1)
                    elif kwargs['sampling'] == 'top_k':
                        target_next = self.top_k_sampling(logits=logits,k=kwargs['k'])
                        target_next = torch.tensor(target_next).unsqueeze(0).unsqueeze(0).to(self.device)
                    elif kwargs['sampling'] == 'top_p':
                        target_next = self.top_p_sampling(logits=logits,p=kwargs['p'])
                        target_next = torch.tensor(target_next).unsqueeze(0).unsqueeze(0).to(self.device)
                    elif kwargs['sampling'] == 'hard':
                        target_next = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
                    elif kwargs['sampling'] == 'repeat_penalty':
                        logits = self.apply_repeat_penalty(logits,target[0][-3:])
                        probs = F.softmax(logits / kwargs['temperature'], dim=-1)
                        target_next = torch.multinomial(probs, num_samples=1).unsqueeze(0) # (B, 1)
                    target = torch.cat((target, target_next), dim=1).to(self.device)
                    c+=1
            return target[0]
           
    def top_k_sampling(self,logits,k=5):
        """
        Perform top-k sampling on the given logits.

        Args:
        logits (numpy.ndarray): Array of logits representing the predicted probabilities.
        k (int): Number of top candidates to consider for sampling.

        Returns:
        selected_token (int): The selected token after top-k sampling.
        """
        logits = logits.cpu().numpy()
        sorted_indices = np.argsort(logits)[::-1]  # Sort in descending order
        top_indices = sorted_indices[:k]
        top_probs = self.stable_softmax(logits[top_indices])
        selected_index = np.random.choice(top_indices, p=top_probs)
        return selected_index

    def top_p_sampling(self,logits, p=0.8):
        logits = logits.cpu().numpy()
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = np.cumsum(sorted_logits)
        
        # Find the smallest set of words whose cumulative probability exceeds p
        sorted_indices = sorted_indices[cumulative_probs > p]
        min_index = sorted_indices[0] if len(sorted_indices) > 0 else len(logits) - 1
        
        # Set probabilities of all words outside the set to 0
        logits[:min_index] = -float('inf')
        probabilities = self.stable_softmax(logits)
        
        # Sample from the modified distribution
        sampled_index = np.random.choice(len(logits), size=1, p=probabilities)[0]
        
        return sampled_index

    @staticmethod     
    def stable_softmax(logits):
        """
        Compute the stabilized softmax of logits.

        Args:
        logits (numpy.ndarray): Array of logits.

        Returns:
        softmax_probs (numpy.ndarray): Stabilized softmax probabilities.
        """
        max_logit = np.max(logits)
        logits_shifted = logits - max_logit
        exp_logits_shifted = np.exp(logits_shifted)
        softmax_probs = exp_logits_shifted / np.sum(exp_logits_shifted)
        return softmax_probs

    def apply_repeat_penalty(self, logits, generated_tokens):
        for token in set(generated_tokens):
            logits[token] = logits[token] - 10
        return logits
    
