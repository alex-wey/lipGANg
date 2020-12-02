import numpy as np
import cv2
import os
import librosa
import scipy
import torch
from torch import nn
from torch.nn import functional as F

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class Discriminator_Model(nn.Module):
    def __init__(self):
        self.learning_rate = 0.001

        self.leakyrelu = nn.LeakyReLU(0.2)
        
        # Face video encoder layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.instanceNorm1 = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=(1,2), padding=2)
        self.instanceNorm2 = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.instanceNorm3 = nn.InstanceNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.instanceNorm4 = nn.InstanceNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.instanceNorm5 = nn.InstanceNorm2d(512)
        self.conv6 = nn.conv2D(filters=512, kernel_size=3, strides=1, padding=0)
        
        # Audio encoder layers
        self.audioConv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.audioInstanceNorm1 = nn.InstanceNorm2d(32)
        self.audioConv2 = nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1)	#27X9
        self.audioInstanceNorm2 = nn.InstanceNorm2d(64)
        self.audioConv3 = nn.Conv2d(64, 128, kernel_size=3, stride=(3, 1), padding=1) 		#9X9
        self.audioInstanceNorm3 = nn.InstanceNorm2d(128)
        self.audioConv4 = nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=1)	#3X3
        self.audioInstanceNorm4 = nn.InstanceNorm2d(256)
        self.audioConv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)	#1X1
        self.audioInstanceNorm5 = nn.InstanceNorm2d(512)
        self.audioConv6 = nn.conv2D(512, 512, kernel_size=1, stride=1, padding=1)
        self.audioInstanceNorm6 = nn.InstanceNorm2d(512)
  
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
	def contrastive_loss(y_true, y_pred):
		margin = 1.
		loss = (1. - y_true) * torch.square(y_pred) + y_true * torch.square(torch.maximum(0., margin - y_pred))
		return torch.mean(loss)

	def forward(self, inputs, labels, mel_step_size):
		############# encoder for face/identity
		input_face = torch.randn((inputs.img_size, inputs.img_size, 3))
		# Converted SAME padding to an int padding variable calculated with k = (n-1)/2
		x = self.leakyrelu(self.instanceNorm1(self.conv1(x)))
		x = self.leakyrelu(self.instanceNorm2(self.conv2(x)))
		x = self.leakyrelu(self.instanceNorm3(self.conv3(x)))
		x = self.leakyrelu(self.instanceNorm4(self.conv4(x)))
		x = self.leakyrelu(self.instanceNorm5(self.conv5(x)))
		x = self.conv6(x)
		face_embedding = torch.flatten(x)
		
		############# encoder for audio
		input_audio = torch.randn((80, mel_step_size, 1))
		x = self.leakyrelu(self.audioInstanceNorm1(self.audioConv1(input_audio)))
		x = self.leakyrelu(self.audioInstanceNorm2(self.audioConv2(x)))
		x = self.leakyrelu(self.audioInstanceNorm3(self.audioConv3(x)))
		x = self.leakyrelu(self.audioInstanceNorm4(self.audioConv4(x)))
		x = self.leakyrelu(self.audioInstanceNorm5(self.audioConv5(x)))
		x = self.leakyrelu(self.audioInstanceNorm6(self.audioConv6(x)))

		audio_embedding = torch.flatten() (x)

		# L2-normalize before taking L2 distance
		l2_normalize = Lambda(lambda x: F.normalize(x, dim=1)) 
		face_embedding = l2_normalize(face_embedding)
		audio_embedding = l2_normalize(audio_embedding)

		d = Lambda(lambda x: torch.sqrt(torch.sum(torch.square(x[0] - x[1]), dim=1, keepdim=True)))([face_embedding,audio_embedding])
		
		return d

if __name__ == '__main__':
	model = create_model()
