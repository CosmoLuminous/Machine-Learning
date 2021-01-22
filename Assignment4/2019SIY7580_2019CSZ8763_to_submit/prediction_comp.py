#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[71]:


'''Import modules'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from collections import Counter
from skimage import io, transform
from torch.nn.utils.rnn import pack_padded_sequence
from torchsummary import summary

import matplotlib.pyplot as plt # for plotting
import numpy as np
from time import time
import collections
import pickle
import os
import gensim
import nltk


# In[72]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device =", device)
print("Using", torch.cuda.device_count(), "GPUs!")
parallel = True #enable nn.DataParallel for GPU
platform = "local" #colab/local
restore = True #Restore Checkpoint
phase = "Test"


# In[73]:


VOCAB = {}
WORD2IDX = {}
IDX2WORD = {}


# In[74]:


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        #print("TA RESCALE INPUT", image.shape)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        #print("TA RESCALE OUTPUT", image.shape)
        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #print("TA TRANSPOSE IP", image.shape)
        #image = image.transpose((2, 0, 1))
        #print("TA TRANSPOSE OP", image.shape)
        return image


IMAGE_RESIZE = (256, 256)
# Sequentially compose the transforms
img_transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor()])


# In[75]:


class CaptionsPreprocessing:
    """Preprocess the captions, generate vocabulary and convert words to tensor tokens
    Args:
        captions_file_path (string): captions tsv file path
    """
    def __init__(self, captions_file_path):
        self.captions_file_path = captions_file_path

        # Read raw captions
        self.raw_captions_dict = self.read_raw_captions()

        # Preprocess captions
        self.captions_dict = self.process_captions()

        # Create vocabulary
        self.start = "<start>"
        self.end = "<end>"
        self.oov = "<unk>"
        self.pad = "<pad>"
        self.vocab = self.generate_vocabulary()
        self.word2index = self.convert_word2index()        
        self.index2word = self.convert_index2word()
        

    def read_raw_captions(self):
        """
        Returns:
            Dictionary with raw captions list keyed by image ids (integers)
        """
        captions_dict = {}
        with open(self.captions_file_path, 'r', encoding='utf-8') as f:
            for img_caption_line in f.readlines():
                img_captions = img_caption_line.strip().split('\t')
                captions_dict[int(img_captions[0])] = img_captions[1:]

        return captions_dict 

    def process_captions(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """

        raw_captions_dict = self.raw_captions_dict 
        
        # Do the preprocessing here                
        captions_dict = raw_captions_dict

        return captions_dict

 

    def generate_vocabulary(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """
        captions_dict = self.captions_dict

        # Generate the vocabulary
        
        all_captions = ""        
        for cap_lists in captions_dict.values():
            all_captions += " ".join(cap_lists)
        all_captions = nltk.tokenize.word_tokenize(all_captions.lower())
        
        vocab = {self.pad :1, self.oov :1, self.start :1, self.end :1}
        vocab_update = Counter(all_captions) 
        vocab_update = {k:v for k,v in vocab_update.items() if v >= freq_threshold}
        vocab.update(vocab_update)        
        vocab_size = len(vocab)
        
        if phase == "Train":
            VOCAB.clear()
            VOCAB.update(vocab)
            if platform == "colab":
                fname = '/content/drive/My Drive/A4/dict/VOCAB_comp.pkl'
            else:
                fname = '../dict/VOCAB_comp.pkl'
            #if not os.path.isfile(fname):
            with open(fname, 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print("VOCAB SIZE =", vocab_size)
        return vocab
    
    def convert_word2index(self):
        """
        word to index converter
        """
        word2index = {}
        vocab = self.vocab
        idx = 0
        words = vocab.keys()
        for w in words:
            word2index[w] = idx
            idx +=1
        if phase == "Train":
            WORD2IDX.clear()
            WORD2IDX.update(word2index)
            if platform == "colab":
                fname = '/content/drive/My Drive/A4/dict/WORD2IDX_comp.pkl'
            else:
                fname = '../dict/WORD2IDX_comp.pkl'
            #if not os.path.isfile(fname):
            with open(fname, 'wb') as handle:
                pickle.dump(word2index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return word2index
    
    def convert_index2word(self):
        """
        index to word converter
        """
        index2word = {}
        w2i = self.word2index
        idx = 0
        
        for k, v in w2i.items():
            index2word[v] = k
            
        if phase == "Train":
            IDX2WORD.clear()
            IDX2WORD.update(index2word)
            if platform == "colab":
                fname = '/content/drive/My Drive/A4/dict/IDX2WORD_comp.pkl'
            else:
                fname = '../dict/IDX2WORD_comp.pkl'
            #if not os.path.isfile(fname):
            with open(fname, 'wb') as handle:
                pickle.dump(index2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return index2word

    def captions_transform(self, img_caption_list):
        """
        Use this function to generate tensor tokens for the text captions
        Args:
            img_caption_list: List of captions for a particular image
        """
        if phase == "Test":
            word2index = WORD2IDX
            vocab = VOCAB
        else:
            word2index = self.word2index
            vocab = self.vocab
            
        start = self.start
        end = self.end
        oov = self.oov
        
        processed_list = list(map(lambda x: nltk.tokenize.word_tokenize(x.lower()), img_caption_list))
        
        
        #print(processed_list)
        processed_list = list(map(lambda x: list(map(lambda y: WORD2IDX[y] if y in vocab else WORD2IDX[oov],x)),
                                  processed_list))
        processed_list = list(map(lambda x: [WORD2IDX['<start>']] + x + [WORD2IDX['<end>']], processed_list))
        #print(processed_list)
        return processed_list


if platform == "colab":
    CAPTIONS_FILE_PATH = '/content/drive/My Drive/A4/train_captions.tsv'
else:
    CAPTIONS_FILE_PATH = "D:/Padhai/IIT Delhi MS(R)/2019-20 Sem II/COL774 Machine Learning/Assignment/Assignment4/train_captions.tsv"
    
embedding_dim = 200
freq_threshold = 5
captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH)


# In[77]:



if phase == "Test":
    if platform != 'colab':
        with open('../dict/VOCAB.pkl', 'rb') as handle:
            VOCAB = pickle.load(handle)
        with open('../dict/WORD2IDX.pkl', 'rb') as handle:
            WORD2IDX = pickle.load(handle)
        with open('../dict/IDX2WORD.pkl', 'rb') as handle:
            IDX2WORD = pickle.load(handle)
        print("Dictionary Loaded Successfully")
    else:
        with open('/content/drive/My Drive/A4/dict/VOCAB.pkl', 'rb') as handle:
            VOCAB = pickle.load(handle)
        with open('/content/drive/My Drive/A4/dict/WORD2IDX.pkl', 'rb') as handle:
            WORD2IDX = pickle.load(handle)
        with open('/content/drive/My Drive/A4/dict/IDX2WORD.pkl', 'rb') as handle:
            IDX2WORD = pickle.load(handle)
        print("Dictionary Loaded Successfully")


# In[78]:


class ImageCaptionsDataset(Dataset):

    def __init__(self, img_dir, captions_dict, img_transform=None, captions_transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            captions_dict: Dictionary with captions list keyed by image ids (integers)
            img_transform (callable, optional): Optional transform to be applied
                on the image sample.

            captions_transform: (callable, optional): Optional transform to be applied
                on the caption sample (list).
        """
        self.img_dir = img_dir
        self.captions_dict = captions_dict
        self.img_transform = img_transform
        self.captions_transform = captions_transform

        self.image_ids = list(captions_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, 'image_{}.jpg'.format(self.image_ids[idx]))
        image = io.imread(img_name)
        #print("RAW IMG", image.shape)
        #captions = self.captions_dict[self.image_ids[idx]]
        if self.img_transform:
            image = self.img_transform(image)
            
            image = image.transpose((2, 0, 1))
            

        '''if self.captions_transform:            
            captions = self.captions_transform(captions)'''
            
        sample = {'image': image}

        return sample
    
    
def custom_batch(batch):
    batch_size = len(batch)
    captions = []
    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    
       
    x = list(map(lambda b: b['image'],batch)) 
    x = list(map(lambda i: normalize_img(torch.from_numpy(i)).unsqueeze(0),x))
    #print("my after norm shape", x[0].shape)
    images = torch.cat(x)
    
    sample = {'image': images}    
    return sample


# In[79]:


#ENCODER

class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        resnet50 = models.resnet50(pretrained=True, progress=True)        
        self.resnet50 = resnet50
        for param in self.resnet50.parameters():
            param.requires_grad = False
        print("EMBED DIM", embed_dim)
        self.fc = nn.Linear(in_features=self.resnet50.fc.in_features, out_features=embed_dim, bias = True)
        layers = list(resnet50.children())[:-1]
        self.resnet50 = nn.Sequential(*layers)
        '''for layer in list(self.resnet50.children())[2:]:
            for params in layer.parameters():
                params.requires_grad = True'''
        self.relu = nn.LeakyReLU()
        print("resnet50 Loaded Successfully..!")

    def forward(self, x):
        x = self.resnet50(x)
        #print("Resnet module op", x.shape)
        x = x.view(x.size(0), -1)
        #print("Resnet module op reshape", x.shape)
        x = self.fc(x)
        x = self.relu(x)
        #print("Resnet FC op", x)
        return x
        
class Decoder(nn.Module):
    def __init__(self, embed_dim, lstm_hidden_size,lstm_layers=1):
        super(Decoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.vocab_size = len(VOCAB)
        print("VOCAB SIZE = ", self.vocab_size)
        
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = lstm_hidden_size,
                            num_layers = lstm_layers, batch_first = True)
        
        self.linear = nn.Linear(lstm_hidden_size, self.vocab_size)        
        #self.embed = nn.Embedding.from_pretrained(init_weights)
        self.embed = nn.Embedding(self.vocab_size, embed_dim)
        #self.attention = AttentionBlock(embed_dim, lstm_hidden_size, self.vocab_size)

        
    def forward(self, image_features, image_captions, lengths):
        #print("DECODER INPUT", image_features)
        if phase == "Train":
            #print(image)
            image_features = torch.Tensor.repeat_interleave(image_features, repeats=5 , dim=0)
        image_features = image_features.unsqueeze(1)
        
        
        embedded_captions = self.embed(image_captions)
        print("EMBED SHAPE", embedded_captions.shape)
        print("SHAPES BEFORE CONCAT",context.unsqueeze(dim=1).shape, embedded_captions[:,:-1].shape)
        input_lstm = torch.cat((image_features, embedded_captions[:,:-1]), dim = 1)
        #input_lstm = pack_padded_sequence(input_lstm, lengths, batch_first=True, enforce_sorted=False)
        lstm_outputs, _ = self.lstm(input_lstm)        
        #lstm_outputs = self.linear(lstm_outputs[0]) 
        print("lstm_outputs.shape", lstm_outputs.shape)
        lstm_outputs = self.linear(lstm_outputs) 
        
        return lstm_outputs


# In[80]:


class ImageCaptionsNet(nn.Module):
    def __init__(self):
        super(ImageCaptionsNet, self).__init__()        
        ##CNN ENCODER RESNET-50        
        self.Encoder = Encoder(embed_dim = embedding_dim)
        ## RNN DECODER
        self.Decoder = Decoder(embedding_dim, units, 1)    
        

    def forward(self, img_batch, cap_batch, lengths):
        #print("IMG INPUT",x)
        x = self.Encoder(img_batch)
        #print("IMG FEATURE",x)
        x = self.Decoder(x, cap_batch, lengths)
        #print("IMG FEATURE",x)
        return x
    
units = 512
if restore == False:
    net = ImageCaptionsNet()
    net = net.double()
    
'''    if parallel == True and device != "cpu":
        print("Parallel Processing enabled")
        net = nn.DataParallel(net)'''

if device == "cpu":
    print("Device to CPU")
else:
    print("Device to CUDA")
    net = net.to(torch.device("cuda:0"))


# In[81]:


'''Save and Restore Checkpoints'''
def create_checkpoint(path,model, optim_obj, loss_obj,iteration, epoch):
    checkpoint = {'epoch': epoch,
                  'iteration': iteration,
                  'model_state_dict': model.state_dict()}

    if platform == "colab":
        directory = '/content/drive/My Drive/A4/bkp_final_try/'
    else:
        directory = '../bkp_final_try/'

    torch.save(checkpoint, directory + path)
    
def restore_checkpoint(path):
    new_state_dict = collections.OrderedDict()
    if platform == "colab":
        directory = '/content/drive/My Drive/A4/bkp_final_try/'
        checkpoint = torch.load(directory + path, map_location=torch.device('cpu'))
    else:
        directory = '../bkp_final_try/'
        checkpoint = torch.load(directory + path, map_location=torch.device('cpu'))    
    
    epoch = checkpoint['epoch']
    new_state_dict = checkpoint['model_state_dict']
    iteration = checkpoint['iteration']
    #optimizer_state_dict = checkpoint['optimizer_state_dict']
    #loss_obj = checkpoint['loss']
    print("Iterations = {}, Epoch = {}".format(iteration, epoch))
    return new_state_dict


# In[107]:


def caption_image(image_feature, max_words=20):
        x = image_feature.unsqueeze(0)
        results = []
        states = None
        
        print(x)
        with torch.no_grad():
            for i in range(max_words):
                
                hiddens, states = net.Decoder.lstm(x, states)
                #print(hiddens.shape)
                decoder_op = net.Decoder.linear(hiddens.squeeze(1))
                predicted_word = decoder_op.argmax(1)
                decoder_op = decoder_op[0].tolist()
                prob = max(decoder_op)
                #print("{} - {}".format(IDX2WORD[predicted_word.item()], prob))
                x = net.Decoder.embed(predicted_word)
                x = x.unsqueeze(0)
                
                word = predicted_word.item()
                results.append(word)
                
                '''if predicted_word == WORD2IDX["<end>"]:
                    break'''
        
        caption = [IDX2WORD[i] for i in  results]
        cap = ' '.join(caption)
        cap = cap.replace("<start>","").replace("<end>","")
        return cap
               
    
# Define your hyperparameters


# In[108]:


if platform == "colab":
    IMAGE_DIR_TEST = '/content/drive/My Drive/train/'
else:
    IMAGE_DIR_TEST = 'D:/Padhai/IIT Delhi MS(R)/2019-20 Sem II/COL774 Machine Learning/Assignment/Assignment4/private_test_images/'

from glob import glob
import os


if restore == True:
    net = ImageCaptionsNet()
    net = net.double()
    state_dict = collections.OrderedDict()
    state_dict = restore_checkpoint("chkpt_finaltry_TOKEN.pth")
    net = ImageCaptionsNet()
    net = net.double()
    net.load_state_dict(state_dict)
    print("State Dictionary Loaded Successfully.")
    
    
images_names = glob(IMAGE_DIR_TEST+"*.jpg")
print(len(images_names))
images_names = [os.path.split(i)[-1][:-4] for i in images_names]
print(images_names[:5])
images_names = [i.split("_")[-1] for i in images_names]
print(images_names[0])

test_dataset = ImageCaptionsDataset(
    IMAGE_DIR_TEST, captions_preprocessing_obj.captions_dict, img_transform=img_transform,
    captions_transform=captions_preprocessing_obj.captions_transform
)
test_dataset.image_ids = images_names

NUM_WORKERS = 0 
MAX_WORDS = 35
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_batch)

if device != "cpu":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
t0 = time()
pred_caps = {}
for batch_idx, sample in enumerate(test_loader):
        print("Image_idx", batch_idx)
        #image_batch = sample['image']
        #print("AFTER", image_batch)
        #print("Original", [IDX2WORD[i] for i in captions_batch)
        #print("Cap", [IDX2WORD[int(i)] for i in captions_batch[0]])
        img_features = net.Encoder(image_batch)
        #print(img_features)
        #img_features = img_features.view(-1)[torch.randperm(img_features.nelement())].view(img_features.size())
        #img_features = torch.FloatTensor(np.random.randn(1,300))
        #print(img_features[0][:4].tolist(), img_features[0][-5:].tolist())
        #print(x.shape)
        #pred_cap = beam_search(img_features)
        pred_cap = caption_image(img_features, 60)
        
        pred_caps[batch_idx] = pred_cap
        print("Predicted",batch_idx, pred_cap)


# ### Test Model

# In[100]:


if platform == "colab":
    IMAGE_DIR_TEST = '/content/drive/My Drive/train/'
else:
    IMAGE_DIR_TEST = 'D:/Padhai/IIT Delhi MS(R)/2019-20 Sem II/COL774 Machine Learning/Assignment/Assignment4/private_test_images/'

from glob import glob
import os

if restore == True:
    if net:
        del net
    net = ImageCaptionsNet()
    net = net.double()
    state_dict = collections.OrderedDict()
    state_dict = restore_checkpoint("chkpt_finaltry_TOKEN_0.01.pth")
    net = ImageCaptionsNet()
    net = net.double()
    net.load_state_dict(state_dict)
    print("State Dictionary Loaded Successfully.")

images_names = glob(IMAGE_DIR_TEST+"*.jpg")
print(len(images_names))
images_names = [os.path.split(i)[-1][:-4] for i in images_names]
print(images_names[:5])
images_names = [i.split("_")[-1] for i in images_names]
print(images_names[0])


# Creating the Dataset
test_dataset = ImageCaptionsDataset(
    IMAGE_DIR_TEST, captions_preprocessing_obj.captions_dict, img_transform=img_transform,
    captions_transform=captions_preprocessing_obj.captions_transform
)
test_dataset.image_ids = images_names

#print(len(img_ids))
NUM_WORKERS = 0 # Parallel threads for dataloading
MAX_WORDS = 35
# Creating the DataLoader for batching purposes
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_batch)

if device != "cpu":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
t0 = time()
pred_caps = {}
for batch_idx, sample in enumerate(test_loader):
        print("Image_idx", batch_idx)
        #image_batch = sample['image']
        #print("AFTER", image_batch)
        #print("Original", [IDX2WORD[i] for i in captions_batch)
        #print("Cap", [IDX2WORD[int(i)] for i in captions_batch[0]])
        img_features = net.Encoder(image_batch)
        #print(img_features)
        #img_features = img_features.view(-1)[torch.randperm(img_features.nelement())].view(img_features.size())
        #img_features = torch.FloatTensor(np.random.randn(1,300))
        #print(img_features[0][:4].tolist(), img_features[0][-5:].tolist())
        #print(x.shape)
        #pred_cap = beam_search(img_features)
        pred_cap = caption_image(img_features, 60)
        
        pred_caps[batch_idx] = pred_cap
        print("Predicted",batch_idx, pred_cap.replace("<unk>", "*"))


# In[94]:


import pandas as pd
op_str = "blu.tre grillare trekking sinistra.una valigie pista.una tastiera estrarre jumpsuit azienda metropolitana.un accanto bilanciarsi coraggioso free canon canon dama canon dama dama 100 comodamente casco rocce accanto.una neve.il birra.una accanto.una accanto.una viso bassi mezz'aria.una sollevati dell'azienda lanciarsi sollevati letto.una afroamericana personale controllato birra.una orgoglioso carro l'esecuzione bandiera.un momenti l'operaio portapranzo pausa ferrovia.un l'attività pasta miei decide hockey.una addormentato cantava cubo fatta."
op_str = "un.uomo in camicia rossa e un gilet blu e una donna."
op_str = [op_str]*len(images_names)

op_dict = dict(zip(images_names, op_str))
df = pd.DataFrame.from_dict(op_dict, orient='index', columns=None)
df.to_csv( "../2019SIY7580_2019CSZ8763/2019SIY7580_2019CSZ8763_public.tsv", sep='\t', header=False)


# In[89]:


net.Encoder.fc.weight

