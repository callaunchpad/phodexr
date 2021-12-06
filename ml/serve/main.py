import sys
sys.path.append("..")

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from transformers import DistilBertTokenizerFast, DistilBertModel
from models.simple_resnet import ResNet50

print('[*] Beginning server startup!')

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='3' # Change this ID to an unused GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Load models
'''

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
distilbert.eval()

print('[*] DistilBERT Loaded')

vision_encoder = ResNet50(num_classes=768).to(device)
vision_encoder.load_state_dict(torch.load('../weights/vision_adam_20ep.pt'))
vision_encoder.eval()

print('[*] Vision Encoder Loaded')

'''
Initialize server and setup CORS
'''

app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''
Util functions
'''

normalize_resize_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

'''
NLP Encoder Route
'''

@app.get('/api/nlp')
async def nlp_encode(text: str):
    print(text)
    # inputs is a dict with input_ids and attention_mask
    tokenized_labels = tokenizer([text], padding="max_length", truncation=True, return_tensors="pt").to(device)
    # run inputs through DistilBERT
    text_embeddings_full = distilbert(**tokenized_labels)
    
    # NOTE: last_hidden_state is (batch_size, sequence_length, hidden_size)
    # we want the CLS token which is the first token in all the sequences
    text_embeddings = text_embeddings_full.last_hidden_state[:,0,:]

    #normalize embeddings
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim = True)

    result = {
        'embedding': text_embeddings.squeeze().tolist()
    }

    return result

'''
Vision Encoder Route
'''

@app.post('/api/vision')
async def vision_encode(image: UploadFile = File(...)):
    try:
        img_contents = await image.read()
        # Convert string data to numpy array
        np_img = np.fromstring(img_contents, np.uint8)
        # Convert numpy array to image
        parsed_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        return {'error': str(e)}

    inputs = normalize_resize_transform(parsed_img).to(device)
    inputs = inputs.unsqueeze_(0)

    image_embeddings = vision_encoder(inputs)

    #normalize embeddings
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim = True)

    result = {
        'embedding': image_embeddings.squeeze().tolist()
    }

    return result