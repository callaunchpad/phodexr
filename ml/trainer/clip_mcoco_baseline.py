import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import random

from transformers import DistilBertTokenizerFast, DistilBertModel
from ml.dataloaders.mcoco import get_cococaptions_dataloader
from ml.models.simple_resnet import ResNet50
from ml.losses.adabound import AdaBound

# setup random seed
random.seed(17)

def accuracy(output, target, k=1):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(1. / batch_size)

def train_clip_mcoco_baseline(epochs, batch_size, learning_rate, optimizer, decay_epoch, vision_weights='', nlp_weights='', unfreeze_nlp=False, debug=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    vision_encoder = ResNet50(num_classes=768).to(device)
    if debug:
        dataset_loader = get_cococaptions_dataloader(mode='debug', batch_size=batch_size)
        wandb.init(project='debug_normalized_clip_mcoco_baseline', entity='phodexrdev')
    else:
        dataset_loader = get_cococaptions_dataloader(mode='train', batch_size=batch_size)
        wandb.init(project='normalized_clip_mcoco_baseline', entity='phodexrdev')

    criterion = nn.CrossEntropyLoss()
    if optimizer == 'sgd':
        vision_optimizer = optim.SGD(vision_encoder.parameters(), lr=learning_rate, momentum=0.9)
        nlp_optimizer = optim.SGD(distilbert.parameters(), lr=learning_rate, momentum=0.9)

        config = wandb.config
        config.update({
            "num_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optim": "sgd",
            "device": device
        })
    elif optimizer == 'adam':
        vision_optimizer = optim.Adam(vision_encoder.parameters(), lr=learning_rate)
        nlp_optimizer = optim.Adam(distilbert.parameters(), lr=learning_rate)

        config = wandb.config
        config.update({
            "num_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optim": "adam",
            "device": device
        })
    elif optimizer == 'adabound':
        vision_optimizer = AdaBound(vision_encoder.parameters(), lr=learning_rate, final_lr=0.1)
        nlp_optimizer = AdaBound(distilbert.parameters(), lr=learning_rate, final_lr=0.1)
        
        config = wandb.config
        config.update({
            "num_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optim": "adabound",
            "device": device
        })

    if vision_weights:
        checkpoint = torch.load(vision_weights)
        if 'model' in checkpoint:
            vision_encoder.load_state_dict(checkpoint['model'])
        else:
            vision_encoder.load_state_dict(checkpoint)
        if 'optim' in checkpoint:
            vision_optimizer.load_state_dict(checkpoint['optim'])
    if nlp_weights:
        if checkpoint['model']:
            distilbert.load_state_dict(checkpoint['model'])
        else:
            vision_encoder.load_state_dict(checkpoint)
        if checkpoint['optim']:
            nlp_optimizer.load_state_dict(checkpoint['optim'])

    vision_scheduler = optim.lr_scheduler.StepLR(vision_optimizer, step_size=decay_epoch, gamma=0.1) # decay lr by 0.1 after x steps (begin hill climbing)
    nlp_scheduler = optim.lr_scheduler.StepLR(nlp_optimizer, step_size=decay_epoch, gamma=0.1) # decay lr by 0.1 after x steps (begin hill climbing)

    # freeze distilbert weights
    if not unfreeze_nlp: # only freeze if unfreeze_nlp is false
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
            if unfreeze_nlp:
                nlp_optimizer.zero_grad()

            # inputs is a dict with input_ids and attention_mask
            tokenized_labels = tokenizer(randomized_labels, padding="max_length", truncation=True, return_tensors="pt").to(device)
            # run inputs through DistilBERT
            text_embeddings_full = distilbert(**tokenized_labels)
            
            # NOTE: last_hidden_state is (batch_size, sequence_length, hidden_size)
            # we want the CLS token which is the first token in all the sequences
            text_embeddings = text_embeddings_full.last_hidden_state[:,0,:]
            image_embeddings = vision_encoder(inputs)

            #noramlize embeddings
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim = True)

            #print(text_embeddings.shape, image_embeddings.shape)
            t_log = nn.Parameter(torch.ones(1), requires_grad = True)
            logits = torch.matmul(text_embeddings, image_embeddings.t()) * torch.exp(t_log).to(device)
            #print(logits.shape)

            # cross entropy loss
            labels = torch.arange(logits.shape[0]).to(device)

            loss1 = criterion(logits,labels)
            loss2 = criterion(torch.transpose(logits,0,1), labels)
            net_loss = (loss1 + loss2)/2

            # TODO: figure out which direction is which :')
            img_top1_acc = accuracy(logits, labels, k=1)
            img_top5_acc = accuracy(logits, labels, k=5)
            text_top1_acc = accuracy(logits.t(), labels, k=1)
            text_top5_acc = accuracy(logits.t(), labels, k=5)

            # pass gradients backwards
            net_loss.backward()

            # optimize vision_encoder gradients
            vision_optimizer.step()
            if unfreeze_nlp:
                nlp_optimizer.step()

            # move schedulers forward
            vision_scheduler.step()
            if unfreeze_nlp:
                nlp_scheduler.step()

            # log training metrics
            metric_dict = {
                'epoch': epoch+1,
                'loss': net_loss,
                'loss_1': loss1,
                'loss_2': loss2,
                'img_top1_acc': img_top1_acc,
                'img_top5_acc': img_top5_acc,
                'text_top1_acc': text_top1_acc,
                'text_top5_acc': text_top5_acc
            }

            wandb.log(metric_dict)

    print('Finished Training CLIP Baseline')

    return distilbert, vision_encoder, vision_optimizer, nlp_optimizer
