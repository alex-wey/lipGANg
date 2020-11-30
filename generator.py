from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Conv2DTranspose, Conv2D, BatchNormalization, \
						Activation, Concatenate, Input, MaxPool2D,\
						UpSampling2D, ZeroPadding2D, Lambda, Add

from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import cv2
import os
import librosa
import scipy
from keras.utils import plot_model
import tensorflow as tf
from keras.utils import multi_gpu_model
from discriminator import contrastive_loss

import torch
from torch import nn
from torch.nn import functional as F

class ModelMGPU(nn.Module)):
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

def conv_block(x, cin, cout, kernel_size, stride, padding, act=True):
	x = nn.Conv2D(in_channels=cin, out_channels=cout, kernel_size=kernel_size, 
					stride=stride, padding=padding)(x)
	x = nn.BatchNorm2d(momentum=.8)(x)
	if act:
		x = nn.ReLU()(x)
	return x

def conv_t_block(x, cin, cout, kernel_size, stride, padding, output_padding=0):
	x = nn.ConvTranspose2d(in_channels=cin, out_channels=cout, kernel_size=kernel_size, 
					stride=stride, padding=padding, output_padding=output_padding)(x)
	x = nn.BatchNorm2d(momentum=.8)(x)
	x = nn.ReLU()(x)
	return x

def create_model(args):
	############# encoder for face/identity
	input_face = Input(shape=(args.img_size, args.img_size, 6), name="input_face")

	identity_mapping = conv_block(input_face, 6, 16, kernel_size=7, stride=1, padding=3) # 96x96
	x1_face = conv_block(identity_mapping, 16, 32, kernel_size=3, stride=2, padding=1) # 48x48
	x2_face = conv_block(x1_face, 32, 64, kernel_size=3, stride=2, padding=1) # 24x24
	x3_face = conv_block(x2_face, 64, 128, kernel_size=3, stride=2, padding=1) # 12x12
	x4_face = conv_block(x3_face, 128, 256, kernel_size=3, stride=2, padding=1) # 6x6
	x5_face = conv_block(x4_face, 256, 512, kernel_size=3, stride=2, padding=1) # 3x3
	x6_face = conv_block(x5_face, 512, 512, kernel_size=3, stride=1, padding=0) # 1x1
	x7_face = conv_block(x6_face, 512, 512, kernel_size=1, stride=1, padding=0)
	# x7_face = conv_block(x6_face, 512, 256, kernel_size=1, stride=1, padding=0)

	############# encoder for audio
	input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio")

	x1_audio = conv_block(input_audio, 1, 32, kernel_size=3, stride=1, padding=1)
	x1_audio = conv_block(x1_audio, 32, 32, kernel_size=3, stride=1, padding=1)
	x1_audio = conv_block(x1_audio, 32, 32, kernel_size=3, stride=1, padding=1)

	x2_audio = conv_block(32, 64, kernel_size=3, stride=(3, 1), padding=1)
	x2_audio = conv_block(x2_audio, 64, 64, kernel_size=3, stride=1, padding=1)
	x2_audio = conv_block(x2_audio, 64, 64, kernel_size=3, stride=1, padding=1)

	x3_audio = conv_block(x2_audio, 64, 128, kernel_size=3, stride=3, padding=1)
	x3_audio = conv_block(x3_audio, 128, 128, kernel_size=3, stride=1, padding=1)
	x3_audio = conv_block(x3_audio, 128, 128, kernel_size=3, stride=1, padding=1)

	x4_audio = conv_block(x3_audio, 128, 256, kernel_size=3, stride=(3, 2), padding=1)
	x4_audio = conv_block(x4_audio, 256, 256, kernel_size=3, stride=1, padding=1)

	x5_audio = conv_block(x4_audio, 256, 512, kernel_size=3, stride=1, padding=0)
	x5_audio = conv_block(x5_audio, 512, 512, kernel_size=1, stride=1, padding=0)

	embedding = torch.cat((x7_face, x5_audio), dim=3)

	############# decoder
	x = conv_block(embedding, 512, 512, kernel_size=1, stride=1, padding=0)
	x = conv_t_block(embedding, 1024, 512, kernel_size=3, stride=1, padding=0) # 3x3
	x = torch.cat((x5_face, x), dim=3)

	x = conv_t_block(x, 1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1) #6x6
	x = torch.cat((x4_face, x), dim=3)

	x = conv_t_block(x, 512, 256, kernel_size=3, stride=2, padding=1, output_padding=1) #12x12
	x = torch.cat((x3_face, x), dim=3)

	x = conv_t_block(x, 320, 128, kernel_size=3, stride=2, padding=1, output_padding=1) #24x24
	x = torch.cat((x2_face, x), dim=3)

	x = conv_t_block(x, 160, 64, kernel_size=3, stride=2, padding=1, output_padding=1) #48x48
	x = torch.cat((x1_face, x), dim=3)

	x = conv_t_block(x, 80, 32, kernel_size=3, stride=1, padding=1) #96x96
	x = torch.cat((identity_mapping, x), dim=3)

	x = conv_block(x, 32, 16, kernel_size=3, strides=1, padding=1) #96x96
	x = conv_block(x, 16, 16, kernel_size=3, strides=1, padding=1) #96x96
	x = Conv2D(in_channels=16, out_channels=3, kernel_size=1, strides=1, padding=1) (x)
	prediction = nn.Sigmoid()(x)
	
	model = Model(inputs=[input_face, input_audio], outputs=prediction)
	model.summary()		
	
	if args.n_gpu > 1:
		model = ModelMGPU(model, args.n_gpu)
		
	model.compile(loss='mae', optimizer=(Adam(lr=args.lr) if hasattr(args, 'lr') else 'adam')) 
	
	return model

def create_model_residual(args, mel_step_size):
	def residual_block(inp, num_filters):
		x = conv_block(inp, num_filters)
		x = conv_block(x, num_filters)

		x = Add()([x, inp])
		x = Activation('relu') (x)

		return x

	############# encoder for face/identity
	input_face = Input(shape=(args.img_size, args.img_size, 6), name="input_face")

	identity_mapping = conv_block(input_face, 32, kernel_size=7) # 96x96

	x1_face = conv_block(identity_mapping, 64, kernel_size=5, strides=2) # 48x48
	x1_face = residual_block(x1_face, 64)
	x1_face = residual_block(x1_face, 64)

	x2_face = conv_block(x1_face, 128, 3, 2) # 24x24
	x2_face = residual_block(x2_face, 128)
	x2_face = residual_block(x2_face, 128)
	x2_face = residual_block(x2_face, 128)

	x3_face = conv_block(x2_face, 256, 3, 2) #12x12
	x3_face = residual_block(x3_face, 256)
	x3_face = residual_block(x3_face, 256)

	x4_face = conv_block(x3_face, 512, 3, 2) #6x6
	x4_face = residual_block(x4_face, 512)
	x4_face = residual_block(x4_face, 512)

	x5_face = conv_block(x4_face, 512, 3, 2) #3x3
	x6_face = conv_block(x5_face, 512, 3, 1, padding='valid')
	x7_face = conv_block(x6_face, 512, 1, 1)

	############# encoder for audio
	input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio")

	x = conv_block(input_audio, 32)
	x = residual_block(x, 32)
	x = residual_block(x, 32)

	x = conv_block(x, 64, strides=3)	#27X9
	x = residual_block(x, 64)
	x = residual_block(x, 64)

	x = conv_block(x, 128, strides=(3, 1)) 		#9X9
	x = residual_block(x, 128)
	x = residual_block(x, 128)

	x = conv_block(x, 256, strides=3)	#3X3
	x = residual_block(x, 256)
	x = residual_block(x, 256)

	x = conv_block(x, 512, strides=1, padding='valid')	#1X1
	x = conv_block(x, 512, 1, 1)

	embedding = Concatenate(axis=3)([x7_face, x])

	############# decoder
	x = conv_t_block(embedding, 512, 3, 3)# 3x3
	x = Concatenate(axis=3) ([x5_face, x]) 

	x = conv_t_block(x, 512) #6x6
	x = residual_block(x, 512)
	x = residual_block(x, 512)
	x = Concatenate(axis=3) ([x4_face, x])

	x = conv_t_block(x, 256) #12x12
	x = residual_block(x, 256)
	x = residual_block(x, 256)
	x = Concatenate(axis=3) ([x3_face, x])

	x = conv_t_block(x, 128) #24x24
	x = residual_block(x, 128)
	x = residual_block(x, 128)
	x = Concatenate(axis=3) ([x2_face, x])

	x = conv_t_block(x, 64) #48x48
	x = residual_block(x, 64)
	x = residual_block(x, 64)
	x = Concatenate(axis=3) ([x1_face, x])

	x = conv_t_block(x, 32) #96x96
	x = Concatenate(axis=3) ([identity_mapping, x])
	x = conv_block(x, 16) #96x96
	x = conv_block(x, 16) #96x96

	x = Conv2D(filters=3, kernel_size=1, strides=1, padding="same") (x)
	prediction = Activation("sigmoid", name="prediction")(x)
	
	model = Model(inputs=[input_face, input_audio], outputs=prediction)
	model.summary()		
	
	if args.n_gpu > 1:
		model = ModelMGPU(model , args.n_gpu)
		
	model.compile(loss='mae', optimizer=(Adam(lr=args.lr) if hasattr(args, 'lr') else 'adam')) 
	
	return model

def create_combined_model(generator, discriminator, args, mel_step_size):
	input_face = Input(shape=(args.img_size, args.img_size, 6), name="input_face_comb")
	input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio_comb")

	fake_face = generator([input_face, input_audio])
	discriminator.trainable = False
	d = discriminator([fake_face, input_audio])

	model = Model([input_face, input_audio], [fake_face, d])
	if args.n_gpu > 1:
		model = ModelMGPU(model , args.n_gpu)

	model.compile(loss=['mae', contrastive_loss], 
					optimizer=(Adam(lr=args.lr) if hasattr(args, 'lr') else 'adam'), 
					loss_weights=[1., .01])

	return model

if __name__ == '__main__':
	model = create_model_residual()
	#plot_model(model, to_file='model.png', show_shapes=True)