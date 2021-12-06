import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from transformers import DistilBertTokenizerFast, DistilBertModel
from ml.models.simple_resnet import ResNet50

normalize_resize_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def test_clip():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    vision_encoder = ResNet50(num_classes=768).to(device)

    checkpoint = torch.load('./ml/weights/vision_adam_20ep.pt')
    vision_encoder.load_state_dict(checkpoint)

    distilbert.eval()
    vision_encoder.eval()

    img = Image.open('./ml/images/demo_2.png').convert('RGB')
    img_norm = normalize_resize_transform(img).to(device)
    img_norm = img_norm.unsqueeze_(0)

    phrases = ['boy on a bike', 'helicopter over the ocean', 'flying a kite']

    with torch.no_grad():
        tokenized_labels = tokenizer(phrases, padding="max_length", truncation=True, return_tensors="pt").to(device)
        text_embeddings_full = distilbert(**tokenized_labels)
        
        text_embeddings = text_embeddings_full.last_hidden_state[:,0,:]
        image_embeddings = vision_encoder(img_norm)
        
        scores = torch.matmul(text_embeddings, torch.transpose(image_embeddings, 0, 1))
        scores_softmax = F.softmax(scores.flatten())

        print(list(zip(phrases, scores_softmax.tolist())))

        return text_embeddings, image_embeddings