import torch
print("Welcome to this Repository!")
if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available!")

