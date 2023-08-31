class Statistic:
    def __init__(self):
        self.count_loss = 0
        self.loss = 0
        self.acc = 0
        self.count_acc = 0
    
    def update_loss(self,new_loss):
        self.loss = (self.loss * self.count_loss + new_loss) / (self.count_loss + 1)
        self.count_loss += 1
    
    def update_acc(self,new_acc,count):
        self.acc = (self.acc * self.count_acc + new_acc * count) / (self.count_acc + count)
        self.count_acc += count

    def reset(self):
        self.count_loss = 0
        self.loss = 0
        self.acc = 0
        self.count_acc = 0