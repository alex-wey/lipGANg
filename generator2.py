import torch
from torch import nn
import math

class conv_block(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        """
        Convolution block w/ forward pass
        """
        super().__init__()
        self.block = nn.Sequential(
                    nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding),
                    nn.BatchNorm2d(cout)
                    )
        self.act = nn.ReLU()
        self.residual = residual

    def forward_pass(self, x):
        out = self.block(x)
        if self.residual:
            out += x
        return self.act(out)

class conv_t_block(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride, padding, output_padding=0):
        """
        Convolution transpose block w/ forward pass
        """
        super().__init__()
        self.block = nn.Sequential(
                    nn.ConvTranspose2d(channel_in, channel_out, kernel_size, stride, padding, output_padding),
                    nn.BatchNorm2d(cout)
                    )
        self.act = nn.ReLU()

    def forward_pass(self, x):
        output = self.block(x)
        return self.act(output)

class Generator_Model(nn.Module):
    def __init__(self):
        """
        The model for the generator network is defined here.
        """
        super(Generator_Model, self).__init__()

        # audio encoder
        self.audio_encoder = nn.Sequential(
            conv_block(1, 32, kernel_size=3, stride=1, padding=1),
            # conv_block(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            # conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            conv_block(64, 128, kernel_size=3, stride=3, padding=1),
            # conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
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
            # conv_block(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_block(32, 64, kernel_size=3, stride=2, padding=1),
            # conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            # conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_block(64, 128, kernel_size=3, stride=2, padding=1),
            # conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_block(128, 256, kernel_size=3, stride=2, padding=1),
            # conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_block(256, 512, kernel_size=3, stride=2, padding=1),
            conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
            
            nn.Sequential(conv_block(512, 512, kernel_size=3, stride=1, padding=0),
            conv_block(512, 512, kernel_size=1, stride=1, padding=0))
            ])

        # face decoder
        self.face_decoder = nn.ModuleList([
            nn.Sequential(conv_block(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(conv_t_block(1024, 512, kernel_size=3, stride=1, padding=0),
            conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_t_block(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            # conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_t_block(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            # conv_block(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(384, 384, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_t_block(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            # conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_t_block(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            # conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(conv_t_block(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            conv_block(64, 64, kernel_size=3, stride=1, padding=1, residual=True))
            ])

        self.output_block = nn.Sequential(conv_block(80, 32, kernel_size=3, stride=1, padding=1),
            nn.conv_block(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs