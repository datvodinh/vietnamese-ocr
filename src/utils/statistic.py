class Statistic:
    def __init__(self):
        self.count = 0
        self.loss = 0
    
    def update_loss(self,new_loss):
        self.loss = (self.loss * self.count + new_loss) / (self.count + 1)
        self.count += 1

    def reset(self):
        self.count = 0
        self.loss = 0