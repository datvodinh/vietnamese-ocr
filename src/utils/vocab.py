import torch

class Vocabulary:
    def __init__(self,data_path=None):
        with open(data_path,"r") as f:
            data = f.read()
        new_data = list(map(lambda i:i.split("\t"),data.split("\n")))
        vocab = []
        for i in range(len(new_data)):
            vocab = list(set(vocab + list(new_data[i][1])))
        vocab = sorted(vocab)
        self.letter_to_idx = {l:i+3 for i,l in enumerate(sorted(vocab))}
        self.idx_to_letter = {i+3:l for i,l in enumerate(sorted(vocab))}
        self.max_tar_len = 0
        for x in new_data:
            if len(x[1]) > self.max_tar_len:
                self.max_tar_len = len(x[1])
        
        self.add_special_token()
        self.target_dict   = {x[0]:self.encode(x[1]).long() for x in new_data}
        self.vocab_size = len(vocab)
        
    def encode(self, s):
        indices = [self.letter_to_idx.get(i, None) for i in s]
        pad_len = self.max_tar_len - len(s) + 1
        return torch.tensor([self.letter_to_idx['<sos>']] + indices + [self.letter_to_idx['<eos>']] + [self.letter_to_idx['<pad>']]*pad_len, dtype=torch.float32)
    def decode(self, idx):
        chars = [self.idx_to_letter[int(i)] for i in idx]
        decoded_chars = [c for c in chars if c not in ['<sos>', '<eos>','<pad>']]
        return "".join(decoded_chars)
    
    def add_special_token(self):
        self.letter_to_idx['<sos>'] = 0
        self.letter_to_idx['<eos>'] = 1
        self.letter_to_idx['<pad>'] = 2
        self.idx_to_letter[0] = '<sos>'
        self.idx_to_letter[1] = '<eos>'
        self.idx_to_letter[2] = '<pad>'


