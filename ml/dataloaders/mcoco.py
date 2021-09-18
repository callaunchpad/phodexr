import torch
import torchvision
import torchvision.transforms as transforms

normalize_resize_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_cococaptions_dataloader(train, batch_size, shuffle=True, transform=normalize_resize_transform):
    # ty oleksii for carrying the data on latte
    if train:
        trainset = torchvision.datasets.CocoCaptions(root='/datasets/coco/data/train2017', annFile='/datasets/coco/data/annotations/captions_train2017.json', transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        return trainloader
    else:
        testset = torchvision.datasets.CocoCaptions(root='/datasets/coco/data/val2017', annFile='/datasets/coco/data/annotations/captions_val2017.json', transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        return testloader