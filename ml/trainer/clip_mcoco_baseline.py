import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import random

from transformers import DistilBertTokenizerFast, DistilBertModel
from ml.dataloaders.mcoco import get_cococaptions_dataloader
from ml.models.simple_resnet import ResNet50

# setup random seed
random.seed(17)

def train_clip_mcoco_baseline(epochs, batch_size, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    vision_encoder = ResNet50(num_classes = 768).to(device)
    dataset_loader = get_cococaptions_dataloader(train=True, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    vision_optimizer = optim.SGD(vision_encoder.parameters(), lr=learning_rate, momentum=0.9)

    wandb.init(project='clip_mcoco_baseline', entity='phodexrdev')
    config = wandb.config
    config.update({
        "num_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device
    })

    # freeze distilbert weights
    for param in distilbert.parameters():
        param.requires_grad = False

    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(dataset_loader, 0):
            # get the inputs; data is a list of [input: image, labels: Tuple[str,...]]
            inputs, labels = data
            inputs = inputs.to(device)

            # NOTE: labels is a list of tuples, each tuple is one set of labels
            rand_idx = random.randint(0, len(labels) - 1)
            randomized_labels = list(labels[rand_idx])

            # zero out gradients of vision optimizer
            vision_optimizer.zero_grad()

            # inputs is a dict with input_ids and attention_mask
            tokenized_labels = tokenizer(randomized_labels, padding="max_length", truncation=True, return_tensors="pt").to(device)
            # run inputs through DistilBERT
            text_embeddings_full = distilbert(**tokenized_labels)
            
            # NOTE: last_hidden_state is (batch_size, sequence_length, hidden_size)
            # we want the CLS token which is the first token in all the sequences
            text_embeddings = text_embeddings_full.last_hidden_state[:,0,:]
            image_embeddings = vision_encoder(inputs)
            # print(text_embeddings.shape, image_embeddings.shape)

            image_embeddings = torch.transpose(image_embeddings, 0, 1)
            #print(text_embeddings.shape, image_embeddings.shape)
            t_log = nn.Parameter(torch.ones(1), requires_grad = True)
            logits = torch.matmul(text_embeddings, image_embeddings) * torch.exp(t_log).to(device)
            #print(logits.shape)

            # cross entropy loss
            labels = torch.arange(logits.shape[0]).to(device)

            loss1 = criterion(logits,labels)
            loss2 = criterion(torch.transpose(logits,0,1), labels)
            net_loss = (loss1 + loss2)/2

            # pass gradients backwards
            net_loss.backward()

            # optimize vision_encoder gradients
            vision_optimizer.step()

            wandb.log({ 'epoch': epoch + 1, 'loss': net_loss})

    print('Finished Training CLIP Baseline')

    return distilbert, vision_encoder
