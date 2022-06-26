'''
srcnn.py

682 - Neural Networks - Project

Sreerag Iyer, Chirag Trasikar

This file contains a superresolution model based on the SRCNN architecture.
SRCNN Link: http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, in_channels=3, f1_size=(9, 9), f1_filters=64, \
                f2_filters=32, f3_size=(5, 5)):
        super().__init__()

        # Define the parameters
        self.pad1 = nn.ReplicationPad2d(4) # Used at test time to 
                                    # avoid shrinking of test image

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=f1_filters, \
                               kernel_size=f1_size)
        
        self.conv2 = nn.Conv2d(in_channels=f1_filters, out_channels=f2_filters, \
                               kernel_size=(1, 1))

        self.pad3 = nn.ReplicationPad2d(2) # Used at test time to 
                                    # avoid shrinking of test image

        self.conv3 = nn.Conv2d(in_channels=f2_filters, out_channels=in_channels, \
                               kernel_size=f3_size)
        #self.conv3 = nn.ConvTranspose2d(in_channels=f2_filters, out_channels=in_channels, \
        #                       kernel_size=f3_size)

        #self.alpha = nn.parameter.Parameter(torch.rand(1, requires_grad=True))

        # Initialize the parameters
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)

    
    def forward(self, x):
        # Forward pass through the convolutional network

        x_pad = None
        if self.training == False:
            x_pad = self.pad1(x)
        else: x_pad = x

        z_1 = self.conv1(x_pad)
        a_1 = F.relu(z_1)
        z_2 = self.conv2(a_1)
        a_2 = F.relu(z_2)

        a_2_pad = None
        if self.training == False:
            a_2_pad = self.pad3(a_2)
        else: a_2_pad = a_2

        z_3 = self.conv3(a_2_pad)
        #z_3 = self.conv3(a_2)

        # Convex combination of input and model output
        #z_4 = torch.sigmoid(self.alpha) * x + (1. - torch.sigmoid(self.alpha)) * z_3

        # Truncated output
        #out = torch.sigmoid(z_4)
        #out = torch.sigmoid(z_3)
        out = torch.clamp(z_3, min=0.0, max=255.)
        
        return out