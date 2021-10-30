import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import random

from transformers import DistilBertTokenizerFast, DistilBertModel
from ml.dataloaders.mcoco import get_cococaptions_dataloader

# setup random seed
random.seed(17)

def train_clip_mcoco_baseline(epochs, batch_size, learning_rate):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
    vision_encoder = None
    dataset_loader = get_cococaptions_dataloader(train=True, batch_size=batch_size)

    # freeze distilbert weights
    for param in distilbert.parameters():
        param.requires_grad = False

    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(dataset_loader, 0):
            # get the inputs; data is a list of [input: image, labels: Tuple[str,...]]
            inputs, labels = data

            # NOTE: labels is a list of tuples, each tuple is one set of labels
            rand_idx = random.randint(0, len(labels) - 1)
            randomized_labels = list(labels[rand_idx])

            # inputs is a dict with input_ids and attention_mask
            inputs = tokenizer(randomized_labels, padding="max_length", truncation=True, return_tensors="pt")
            # run inputs through DistilBERT
            text_embeddings_full = distilbert(**inputs)
            # NOTE: last_hidden_state is (batch_size, sequence_length, hidden_size)
            # we want the CLS token which is the first token in all the sequences
            text_embeddings = text_embeddings_full.last_hidden_state[:,0,:]
            break
        break

    print('Finished Training CLIP Baseline')