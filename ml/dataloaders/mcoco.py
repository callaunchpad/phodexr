import torch
import torchvision
import torchvision.transforms as transforms

normalize_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_cococaptions_dataloader(train):
    if train:
        
    else:
