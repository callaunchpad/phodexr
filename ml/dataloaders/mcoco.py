import torch
import torchvision
import torchvision.transforms as transforms

normalize_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_cococaptions_dataloader(train, batch_size, shuffle=True, transform=normalize_transform):
    if train:
        trainset = torchvision.datasets.CocoCaptions(root='/datasets/', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        return trainloader
    else:
        testset = torchvision.datasets.CocoCaptions(root='/datasets/', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        return testloader