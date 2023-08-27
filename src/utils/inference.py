from src.model.model import OCRModel
from src.utils.vocab import Vocabulary
import torch
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Inference:
    def __init__(self,DATA_PATH,TARGET_PATH):
        data_dict = torch.load(DATA_PATH)
        self.model = OCRModel(data_dict['config'],data_dict['vocab_size'])
        self.model.load_state_dict(data_dict['state_dict'])
        self.model.eval()
        self.vocab = Vocabulary(device,TARGET_PATH)
    
    @torch.no_grad()
    def predict(self,src,type='hard'):
        '''
        Generate predictions using the model.
        
        Args:
            src (torch.Tensor): Input source tensor.
            type (str, optional): Type of prediction to generate ('hard' or 'soft').
        
        Returns:
            target (torch.Tensor): Generated target tensor.
        '''
        c = 0
        target = torch.tensor([[0]]).long().to(device) # <sos>
        while target[0][-1] != 1 and c < 20: # <eos>
            logits = self.model(src,target)
            logits = logits[-1,:]
            probs = F.softmax(logits, dim=-1)
            if type=='hard':
                target_next = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
            else:
                target_next = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            target = torch.cat((target, target_next), dim=1).to(device)
            c+=1
        return target

