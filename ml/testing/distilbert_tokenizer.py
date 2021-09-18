import random
from transformers import DistilBertTokenizerFast

from ml.dataloaders.mcoco import get_cococaptions_dataloader

# setup random seed
random.seed(17)

def test_distilbert_tokenizer(epochs, batch_size):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    dataset_loader = get_cococaptions_dataloader(train=True, batch_size=batch_size)

    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(dataset_loader, 0):
            # get the inputs; data is a list of [input: image, labels: Tuple[str,...]]
            inputs, labels = data

            randomized_labels = []
            for label in labels:
                rand_idx = random.randint(0, len(label) - 1)
                randomized_labels.append(label[rand_idx])

            # inputs is a dict with input_ids and attention_mask
            inputs = tokenizer(randomized_labels, padding="max_length", truncation=True)
            print(inputs)
            break
        break

    print('Finished Testing DistilBERT Tokenizer')