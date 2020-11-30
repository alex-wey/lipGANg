from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Conv2D, Conv3D, BatchNormalization, Activation, \
						Concatenate, AvgPool2D, Input, MaxPool2D, UpSampling2D, Add, \
						, 
from keras_contrib.layers import InstanceNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import cv2
import os
import librosa
import scipy
from keras.utils import plot_model
from keras.utils import multi_gpu_model
import torch
from torch import nn
from torch.nn import functional as F

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
class ModelMGPU(nn.Module):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

def contrastive_loss(y_true, y_pred):
	margin = 1.
	loss = (1. - y_true) * torch.square(y_pred) + y_true * torch.square(torch.maximum(0., margin - y_pred))
	return torch.mean(loss)

def conv_block(x, in_channels, out_channels, kernel_size=3, stride=2, padding):
    x = nn.conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)(x)
    
    # Normalization
	x = nn.InstanceNorm2d(num_features=out_channels)(x) 
  
    x = nn.LeakyReLU(.2)(x)
    return x

def create_model(args, mel_step_size):
	############# encoder for face/identity
	input_face = torch.randn((args.img_size, args.img_size, 3))

    # Converted SAME padding to an int padding variable calculated with k = (n-1)/2
    x = conv_block(input_face, 3, 64, kernel_size=7, stride=1, padding=3)
    x = conv_block(x, 64, 128, kernel_size=5, stride=(1,2), padding=2)
    x = conv_block(x, 128, 256, kernel_size=3, stride=2, padding=1)
    x = conv_block(x, 256, 512, kernel_size=3, stride=2, padding=1)
    x = conv_block(x, 512, 512, kernel_size=3, stride=2, padding=1)
    x = nn.conv2D(filters=512, kernel_size=3, strides=1, padding=0)(x)
    face_embedding = torch.flatten(x)
    
	############# encoder for audio
	input_audio = torch.randn((80, mel_step_size, 1))

	x = conv_block(input_audio, 1, 32, stride=1, padding=1)
	x = conv_block(x, 32, 64, stride=3, padding=1)	#27X9
	x = conv_block(x, 64, 128, stride=(3, 1), padding=1) 		#9X9
	x = conv_block(x, 128, 256, stride=3, padding=1)	#3X3
	x = conv_block(x, 256, 512, stride=1, padding=0)	#1X1
	x = conv_block(x, 512, 512, kernel_size=1, stride=1, padding=1)

	audio_embedding = torch.flatten() (x)

	# L2-normalize before taking L2 distance
	l2_normalize = Lambda(lambda x: F.normalize(x, dim=1)) 
	face_embedding = l2_normalize(face_embedding)
	audio_embedding = l2_normalize(audio_embedding)

	d = Lambda(lambda x: torch.sqrt(torch.sum(torch.square(x[0] - x[1]), dim=1, keepdim=True))) ([face_embedding,
																		audio_embedding])

	model = Model(inputs=[input_face, input_audio], outputs=[d])

	model.summary()

	if args.n_gpu > 1:
		model = ModelMGPU(model , args.n_gpu)
		
	model.compile(loss=contrastive_loss, optimizer=Adam(lr=args.lr)) 
	
	return model

if __name__ == '__main__':
	model = create_model()
