from src.model.model import OCRTransformerModel
from src.utils.transform import Transform
import torch
import torch.nn.functional as F
import re
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Inference:
    def __init__(self,MODEL_PATH):
        data_dict = torch.load(MODEL_PATH)
        self.model = OCRTransformerModel(data_dict['config'],data_dict['vocab_size'])
        self.model.load_state_dict(data_dict['state_dict'])
        self.model.eval()  
        self.letter_to_idx = data_dict['letter_to_idx']
        self.idx_to_letter = data_dict['idx_to_letter']
        self.transform = Transform('sobel')

    def predict(self,img,**kwargs):
        src = self.transform(img).unsqueeze(0).to(device)
        target = torch.tensor([[0]]).long().to(device) # <sos>
        target = self.model(src,target,mode='predict',**kwargs)
        return self._remove_repetition(self.decode(target))
    def encode(self, s):
        indices = [self.letter_to_idx.get(i, None) for i in s]
        pad_len = self.max_tar_len - len(s) + 1
        return torch.tensor([self.letter_to_idx['<sos>']] + indices + [self.letter_to_idx['<eos>']] + [self.letter_to_idx['<pad>']]*pad_len, dtype=torch.float32)
    def decode(self, idx):
        chars = [self.idx_to_letter[int(i)] for i in idx]
        decoded_chars = [c for c in chars if c not in ['<sos>', '<eos>','<pad>']]
        return "".join(decoded_chars)
    
    @staticmethod
    def _remove_repetition(input_string):
        return re.sub(r'(.{2,})\1+', r'\1', input_string)