import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from ml.dataloaders.cifar10 import get_cifar10_dataloaders
from ml.models.simple_cnn import SimpleCNN

def train_cnn_cifar10(epochs, batch_size, learning_rate):
    # setup wandb
    wandb.init(project='simplecnn_cifar10', entity='phodexrdev')
    config = wandb.config
    config.update({
        "num_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })

    dataset_loader = get_cifar10_dataloaders(train=True, batch_size=batch_size)
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataset_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics and log to wandb
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1000))

                # log loss to wandb
                wandb.log({ 'epoch': epoch + 1, 'loss': running_loss / 1000})
                running_loss = 0.0

    print('Finished Training')
    return model