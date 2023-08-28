from src.model.model import OCRTransformerModel
from src.utils.vocab import Vocabulary
import torch
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Inference:
    def __init__(self,DATA_PATH,TARGET_PATH):
        data_dict = torch.load(DATA_PATH)
        self.model = OCRTransformerModel(data_dict['config'],data_dict['vocab_size'])
        self.model.load_state_dict(data_dict['state_dict'])
        self.model.eval()  
        self.vocab = Vocabulary(device,TARGET_PATH)
    
    @torch.no_grad()
    def predict(self,src):
        target = torch.tensor([[0]]).long().to(device) # <sos>
        target = self.model(src,target,mode='predict')
        return self.vocab.decode(target)

