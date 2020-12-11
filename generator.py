import torch
from torch import nn
import math

class conv_block(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride, padding, residual=False):
        """
        Convolution block w/ forward pass
        """
        super().__init__()
        self.block = nn.Sequential(
                    nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding),
                    nn.BatchNorm2d(channel_out)
                    )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        return self.act(self.block(x))

class conv_t_block(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride, padding, output_padding=0):
        """
        Convolution transpose block w/ forward pass
        """
        super().__init__()
        self.block = nn.Sequential(
                    nn.ConvTranspose2d(channel_in, channel_out, kernel_size, stride, padding, output_padding),
                    nn.BatchNorm2d(channel_out)
                    )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.block(x))

class Generator_Model(nn.Module):
    def __init__(self):
        """
        The model for the generator network is defined here.
        """
        super(Generator_Model, self).__init__()

        # audio encoder
        self.audio_encoder = nn.Sequential(
            conv_block(1, 32, kernel_size=1, stride=1, padding=1),
            conv_block(32, 32, kernel_size=1, stride=1, padding=1, residual=True),
            
            conv_block(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(64, 128, kernel_size=3, stride=3, padding=1),
            conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(256, 512, kernel_size=3, stride=1, padding=0),
            conv_block(512, 512, kernel_size=1, stride=1, padding=0)
            )

        # face encoder
        self.face_encoder = nn.ModuleList([
            nn.Sequential(conv_block(6, 16, kernel_size=7, stride=1, padding=3)),

            nn.Sequential(conv_block(16, 32, kernel_size=3, stride=2, padding=1),
            conv_block(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_block(32, 64, kernel_size=3, stride=2, padding=1),
            conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_block(64, 128, kernel_size=3, stride=2, padding=1),
            conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_block(128, 256, kernel_size=3, stride=2, padding=1),
            conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_block(256, 512, kernel_size=3, stride=2, padding=1),
            conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
            
            nn.Sequential(conv_block(512, 512, kernel_size=3, stride=1, padding=0),
            conv_block(512, 512, kernel_size=1, stride=1, padding=0))
            ])

        # face decoder
        self.face_decoder = nn.ModuleList([
            nn.Sequential(conv_block(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(conv_t_block(512, 512, kernel_size=3, stride=1, padding=0),
            conv_block(512, 1024, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_t_block(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv_block(512, 768, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_t_block(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv_block(384, 512, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_t_block(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv_block(256, 320, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_t_block(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            conv_block(128, 160, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_t_block(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True))
            ])

        # output blocks for face prediction
        self.pred_output = nn.Sequential(conv_block(64, 32, kernel_size=3, stride=1, padding=1),
            conv_block(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.5, 0.999))

    # performs foward pass on model during training
    def forward(self, audio_seq, face_seq):
        
        audio_embedding = self.audio_encoder(audio_seq)

        features = []
        output = face_seq
        for block in self.face_encoder:
            output = block(output)
            features.append(output)

        output = audio_embedding
        for block in self.face_decoder:
            output = block(output)
            try:
                output = torch.cat((output, features[-1]), dim=1)
            except Exception as e:
                pass

            features.pop()

        outputs = self.pred_output(output)
            
        return outputs
 
