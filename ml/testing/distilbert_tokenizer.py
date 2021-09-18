from transformers import DistilBertTokenizerFast

from ml.dataloaders.mcoco import get_cococaptions_dataloader

def test_distilbert_tokenizer(epochs, batch_size):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    dataset_loader = get_cococaptions_dataloader(train=True, batch_size=batch_size)

    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(dataset_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #print(inputs, labels)
        
            inputs = tokenizer(labels, padding="max_length", truncation=True)
            print(inputs)

    print('Finished Testing DistilBERT Tokenizer')