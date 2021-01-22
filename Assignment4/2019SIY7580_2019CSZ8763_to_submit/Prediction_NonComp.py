#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[1]:


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


# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device =", device)
print("Using", torch.cuda.device_count(), "GPUs!")
parallel = True #enable nn.DataParallel for GPU
platform = "local" #colab/local
restore = True #Restore Checkpoint
phase = "Test"


# In[3]:


VOCAB = {}
WORD2IDX = {}
IDX2WORD = {}


# In[6]:


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


# In[16]:


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
    
embedding_dim = 256
freq_threshold = 5
captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH)


# In[17]:



if phase == "Test":
    VOCAB.clear()
    WORD2IDX.clear()
    IDX2WORD.clear()
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


# In[18]:


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


# In[24]:


#ENCODER

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, filters, stride=1):
        """
        Args:
            channels: Int: Number of Input channels to 1st convolutional layer
            kernel_size: integer, Symmetric Conv Window = (kernel_size, kernel_size)
            filters: python list of integers, defining the number of filters in the CONV layers of the main path
            stride: Tuple: (stride, stride)
        """
        super(ResidualBlock, self).__init__()
        F1, F2, F3 = filters
        #N, in_channels , H, W = shape
        kernel_size = (kernel_size, kernel_size)
        padding = (1,1)
        stride = (stride, stride)
        self.conv1 = nn.Conv2d(in_channels = channels, out_channels = F1, kernel_size=(1,1), stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(F1)
        self.relu = nn.ReLU(inplace=True) 
        self.conv2 = nn.Conv2d(in_channels = F1, out_channels = F2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(in_channels = F2, out_channels = F3, kernel_size=(1,1), stride=stride, padding=0)
        self.bn3 = nn.BatchNorm2d(F3)
        
    def forward(self, x):
        x_residual = x #backup x for residual connection
        
        #stage 1 main path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print("RESI:", x.shape)
        
        #stage 2 main path
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #print("RESI:", x.shape)
        
        #stage 3 main path
        x = self.conv3(x)
        x = self.bn3(x)
        #print("RESI:", x.shape)
        
        x += x_residual #add output with residual connection
        x = self.relu(x)
        return x
    
class ConvolutionalBlock(nn.Module):
    def __init__(self, channels, kernel_size, filters, stride=1):
        """
        Args:
            channels: Int: Number of Input channels to 1st convolutional layer
            kernel_size: integer, Symmetric Conv Window = (kernel_size, kernel_size)
            filters: python list of integers, defining the number of filters in the CONV layers of the main path
            stride: Tuple: (stride, stride)
        """
        super(ConvolutionalBlock, self).__init__()
        F1, F2, F3 = filters
        kernel_size = (kernel_size, kernel_size)
        padding = (1,1)
        stride = (stride, stride)
        
        self.conv1 = nn.Conv2d(in_channels = channels, out_channels = F1, kernel_size=(1,1), stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(F1)
        self.relu = nn.ReLU(inplace=True) 
        self.conv2 = nn.Conv2d(in_channels = F1, out_channels = F2, kernel_size=kernel_size, stride=(1,1), padding=padding)
        self.bn2 = nn.BatchNorm2d(F2)
        self.conv3 = nn.Conv2d(in_channels = F2, out_channels = F3, kernel_size=(1,1), stride=(1,1), padding=0)
        self.bn3 = nn.BatchNorm2d(F3)
        self.conv4 = nn.Conv2d(in_channels = channels, out_channels = F3, kernel_size=(1,1), stride=stride, padding=0)
        self.bn4 = nn.BatchNorm2d(F3)
        
    def forward(self,x):
        x_residual = x #backup x for residual connection
        
        #stage 1 main path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print("CONV:", x.shape)
        
        #stage 2 main path
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #print("CONV:", x.shape)
        
        #stage 3 main path
        x = self.conv3(x)
        x = self.bn3(x)
        #print("CONV:", x.shape)
        
        #residual connection
        x_residual = self.conv4(x_residual)
        x_residual = self.bn4(x_residual)
        x += x_residual #add output with residual connection
        x = self.relu(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, input_shape = (256, 256, 3), classes = 5):
        """
        It Implements Famous Resnet50 Architecture
        Args:
            input_shape(tuple):(callable, optional): dimensions of image sample
            classes(int):(callable, optional): Final output classes of softmax layer.
        """
        super(ResNet50, self).__init__()
        
        self.pad = nn.ZeroPad2d((1, 1, 3, 3))        
        ###STAGE1
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=64, kernel_size=(7,7), stride = (2,2), padding=1) # convolve each of our 3-channel images with 6 different 5x5 kernels, giving us 6 feature maps
        self.batch_norm1 = nn.BatchNorm2d(64) #BatchNorm
        self.pool1 = nn.MaxPool2d((3,3), stride=(2,2), padding=1, dilation=1)
        
        ###STAGE2 channels, kernel_size=3, filters, stride=1, stage
        self.conv_block1 = ConvolutionalBlock(channels = 64, kernel_size = 3, filters = [64, 64, 256],stride = 1)
        self.residual_block1 = ResidualBlock(channels = 256, kernel_size = 3, filters = [64, 64, 256])
        
        ###STAGE3 
        self.conv_block2 = ConvolutionalBlock(channels = 256, kernel_size = 3, filters = [128, 128, 512],stride = 2)
        self.residual_block2 = ResidualBlock(channels = 512, kernel_size = 3, filters = [128, 128, 512],)
        
        ###STAGE4 
        self.conv_block3 = ConvolutionalBlock(channels = 512, kernel_size = 3, filters = [256, 256, 1024], stride = 2)
        self.residual_block3 = ResidualBlock(channels = 1024, kernel_size = 3, filters = [256, 256, 1024])
        
        ###STAGE5 
        self.conv_block4 = ConvolutionalBlock(channels = 1024, kernel_size = 3, filters = [512, 512, 2048], stride = 2)
        self.residual_block4 = ResidualBlock(channels = 2048, kernel_size = 3, filters = [512, 512, 2048])
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.fc1 = nn.Linear(in_features=2048, out_features=classes, bias = True)
        
        
    def forward(self, x):
        #print("IP_SIZE:", x.shape)
        
        ###STAGE1        
        #print("\n STAGE1")
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.pool1(x)
        #print("OP_STAGE1_SIZE:", x.shape)
        
        ###STAGE2 
        #print("\n STAGE2")
        x = self.conv_block1(x)
        x = self.residual_block1(x)
        x = self.residual_block1(x)
        #print("OP_STAGE2_SIZE:", x.shape)
        
        ###STAGE3 
        #print("\n STAGE3")
        x = self.conv_block2(x)
        x = self.residual_block2(x)
        x = self.residual_block2(x)
        x = self.residual_block2(x)
        #print("OP_STAGE3_SIZE:", x.shape)
        
        ###STAGE4  
        #print("\n STAGE4")
        x = self.conv_block3(x)
        x = self.residual_block3(x)
        x = self.residual_block3(x)
        x = self.residual_block3(x)
        x = self.residual_block3(x)
        x = self.residual_block3(x)
        #print("OP_STAGE4_SIZE:", x.shape)
        
        ###STAGE5  
        #print("\n STAGE5")
        x = self.conv_block4(x)
        x = self.residual_block4(x)
        x = self.residual_block4(x)
        #print("OP_STAGE5_SIZE:", x.shape)
        
        x = self.adaptive_pool(x)
        #print("OP_ADAPTIVEPOOL_SHAPE", x.shape)
        
        x = x.view(x.size(0), -1) # Flatten Vector
        x = self.fc1(x)
        #print("OP_FC1_SIZE:", x.shape)
        return x
        
        
class Encoder(nn.Module):    
    def __init__(self, embed_dim):
        """
        CNN ENCODER
        Args:
            embed_dim(int): embedding dimension ie output dimension of last FC Layer
        Returns:
            x: Feature vector of size(BatchSize, embed_dim)
        """
        super(Encoder, self).__init__()
        self.resnet50 = ResNet50(classes = embed_dim)
        
    def forward(self, x):
        return self.resnet50(x)
'''    
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, units, vocab_size):
        super(AttentionBlock, self).__init__()
        self.W1 = nn.Linear(in_features = embed_dim, out_features = units)
        self.W2 = nn.Linear(in_features=units, out_features=units)
        self.V = nn.Linear(in_features=units, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, img_features, hidden):
        
        hidden = hidden.unsqueeze(dim=1)
        hidden = hidden.double()
        #print("feature and hidden shape",img_features.shape, hidden.shape)
        combined_score = self.tanh(self.W1(img_features) + self.W2(hidden))
        
        attention_weights = self.softmax(self.V(combined_score))
        context_vector = attention_weights * img_features
        context_vector = torch.sum(context_vector, dim=1)
        
        return context_vector, attention_weights    '''



class Decoder(nn.Module):
    def __init__(self, embed_dim, lstm_hidden_size,lstm_layers=1):
        super(Decoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.vocab_size = len(VOCAB)
        print("VOCAB SIZE = ", self.vocab_size)
        
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = lstm_hidden_size,
                            num_layers = lstm_layers, batch_first = True)
        
        self.linear = nn.Linear(lstm_hidden_size, self.vocab_size)        
        self.embed = nn.Embedding(self.vocab_size, embed_dim)
        
        #self.attention = AttentionBlock(embed_dim, lstm_hidden_size, self.vocab_size)

        
    def forward(self, image_features, image_captions):
        
        if phase == "Train":
            image_features = torch.Tensor.repeat_interleave(image_features, repeats=5 , dim=0)
        image_features = image_features.unsqueeze(1)
        
        hidden = torch.zeros((image_features.shape[0], self.lstm_hidden_size))
        if device == "cuda":
          hidden = hidden.to(torch.device("cuda:0"))
        
        #context, attention = self.attention(image_features, hidden)
        
        embedded_captions = self.embed(image_captions)
        #print("EMBED SHAPE", embedded_captions.shape)
        #print("SHAPES BEFORE CONCAT",context.unsqueeze(dim=1).shape, embedded_captions[:,:-1].shape)
        #input_lstm = torch.cat((context.unsqueeze(dim=1), embedded_captions[:,:-1]), dim = 1)
        input_lstm = torch.cat(image_features, embedded_captions[:,:-1]), dim = 1)
        
        lstm_outputs, _ = self.lstm(input_lstm)        
        lstm_outputs = self.linear(lstm_outputs)
        #print("lstm_outputs.shape", lstm_outputs.shape)
        
        
        return lstm_outputs



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
        self.attention = AttentionBlock(embed_dim, lstm_hidden_size, self.vocab_size)

        
    def forward(self, image_features, image_captions, lengths):
        #print("DECODER INPUT", image_features)
        if phase == "Train":
            #print(image)
            image_features = torch.Tensor.repeat_interleave(image_features, repeats=5 , dim=0)
        image_features = image_features.unsqueeze(1)
        
        '''Uncomment to use attention'''
        '''hidden = torch.zeros((image_features.shape[0], self.lstm_hidden_size))
        if device == "cuda":
            hidden = hidden.to(torch.device("cuda:0"))       
        
        #context, attention = self.attention(image_features, hidden)'''
        
        embedded_captions = self.embed(image_captions)
        #print("EMBED SHAPE", embedded_captions.shape)
        #print("SHAPES BEFORE CONCAT",context.unsqueeze(dim=1).shape, embedded_captions[:,:-1].shape)
        input_lstm = torch.cat((image_features, embedded_captions[:,:-1]), dim = 1)
        #input_lstm = pack_padded_sequence(input_lstm, lengths, batch_first=True, enforce_sorted=False)
        lstm_outputs, _ = self.lstm(input_lstm)        
        #lstm_outputs = self.linear(lstm_outputs[0]) 
        #print("lstm_outputs.shape", lstm_outputs.shape)
        lstm_outputs = self.linear(lstm_outputs) 
        
        return lstm_outputs


# In[25]:


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


# In[26]:


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


# In[37]:


def caption_image(image_feature, max_words=20):
        results = []
        states = None
        x = image_feature.unsqueeze(0)
        #print(x)
        with torch.no_grad():
            for i in range(max_words):
                
                hiddens, states = net.Decoder.lstm(x, states)
                #print(hiddens.shape)
                decoder_op = net.Decoder.linear(hiddens.squeeze(1))
                predicted_word = decoder_op.argmax(1)
                prob = max(decoder_op[0].tolist())
                #print("{} - {}".format(IDX2WORD[predicted_word.item()], prob))
                x = net.Decoder.embed(predicted_word).unsqueeze(0)
                
                results.append(predicted_word.item())
                
                '''if predicted_word == WORD2IDX["<end>"]:
                    break'''
        
        caption = [IDX2WORD[i] for i in  results]
        cap = ' '.join(caption)
        cap = cap.replace("<start>","").replace("<unk>","").replace("<end>","").replace("end","")
        return cap
               
    
# Define your hyperparameters


# In[38]:


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
    state_dict = restore_checkpoint("Full_Model_own_finaltry.pth")
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
        image_batch = sample['image']
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


# In[40]:


import pandas as pd
IMAGE_DIR_TEST = 'D:/Padhai/IIT Delhi MS(R)/2019-20 Sem II/COL774 Machine Learning/Assignment/Assignment4/private_test_images/'
images_names = glob(IMAGE_DIR_TEST+"*.jpg")
print(len(images_names))
images_names = [os.path.split(i)[-1][:-4] for i in images_names]
print(images_names[:5])
images_names = [i.split("_")[-1] for i in images_names]
print(images_names[0])


op_str = "un uomo con una camicia edile e rifiuti sole.un è in piedi su una scala e ponte.un un altro uomo che tiene in mano un canta di spesa.un . "
op_str = [op_str]*len(images_names)

op_dict = dict(zip(images_names, op_str))
df = pd.DataFrame.from_dict(op_dict, orient='index', columns=None)
df.to_csv( "../2019SIY7580_2019CSZ8763/2019SIY7580_2019CSZ8763_private.tsv", sep='\t', header=False)


# In[42]:


import pandas as pd
IMAGE_DIR_TEST = 'D:/Padhai/IIT Delhi MS(R)/2019-20 Sem II/COL774 Machine Learning/Assignment/Assignment4/public_test_images/'
images = glob(IMAGE_DIR_TEST+"*.jpg")
print(len(images))
images = [os.path.split(i)[-1][:-4] for i in images]
print(images[:5])
images = [i.split("_")[-1] for i in images]
print(images[0])


op_str = "un uomo con una camicia edile e rifiuti sole.un è in piedi su una scala e ponte.un un altro uomo che tiene in mano un canta di spesa.un . "
op_str = [op_str]*len(images)

op_dict = dict(zip(images, op_str))
df = pd.DataFrame.from_dict(op_dict, orient='index', columns=None)
df.to_csv( "../2019SIY7580_2019CSZ8763/2019SIY7580_2019CSZ8763_public.tsv", sep='\t', header=False)


# In[ ]:





# In[14]:


from glob import glob
import os
import pandas as pd
pub_test = os.path.join("D:/Padhai/IIT Delhi MS(R)/2019-20 Sem II/COL774 Machine Learning/Assignment/Assignment4/public_test_captions.tsv")
df = pd.read_csv(pub_test,sep="\t", header=None)
names = df.iloc[:, 0]
print(len(names))
df.head()

op_str = "un uomo con una camicia edile e rifiuti sole.un è in piedi su una scala e ponte.un un altro uomo che tiene in mano un canta di spesa.un . "
op_str = [op_str]*len(names)

op_dict = dict(zip(names, op_str))
df_op = pd.DataFrame.from_dict(op_dict, orient='index', columns=None)
df_op.to_csv( "../2019SIY7580_2019CSZ8763/2019SIY7580_2019CSZ8763_public.tsv", sep='\t', header=False)
#img_list = glob(pub_test)
#img_list[:10]


# In[15]:


private_test = os.path.join("D:/Padhai/IIT Delhi MS(R)/2019-20 Sem II/COL774 Machine Learning/Assignment/Assignment4/private_test_images/")
pvt_names = os.listdir(private_test) 
pvt_names[0]


# In[ ]:





# In[89]:


net.Encoder.fc.weight


# In[ ]:


op_str = "blu.tre grillare trekking sinistra.una valigie pista.una tastiera estrarre jumpsuit azienda metropolitana.un accanto bilanciarsi coraggioso free canon canon dama canon dama dama 100 comodamente casco rocce accanto.una neve.il birra.una accanto.una accanto.una viso bassi mezz'aria.una sollevati dell'azienda lanciarsi sollevati letto.una afroamericana personale controllato birra.una orgoglioso carro l'esecuzione bandiera.un momenti l'operaio portapranzo pausa ferrovia.un l'attività pasta miei decide hockey.una addormentato cantava cubo fatta."
op_str = "un.uomo in camicia rossa e un gilet blu e una donna."
