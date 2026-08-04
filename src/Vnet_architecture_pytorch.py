# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:59:16 2023

@author: ljeantet
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class SpatialDropout1D(nn.Module):
    def __init__(self, p: float = 0.5):
        """
        Spatial Dropout for 1D inputs (batch_size, channels, sequence_length).
        It drops entire channels (features) across all timesteps.

        Args:
            p (float): dropout probability (default=0.5)
        """
        super(SpatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        x = x.unsqueeze(3)  # Add fake height dimension (N, C, L, 1)
        x = self.dropout(x)  # Apply Dropout2d
        return x.squeeze(3)

class VNet4Levels(nn.Module):
    def __init__(self, in_channels, out_channels, model_depth, dropout_rate=0.5, kernel_size=5):
        super(VNet4Levels, self).__init__()

        self.in_channels = in_channels
        self.out_channels=out_channels
        self.depth = model_depth
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size

        self.dropout = SpatialDropout1D(p=dropout_rate)

        # Define layers
        self.conv1 = self.double_conv(self.in_channels, self.depth)
        self.down1 = nn.Conv1d(self.depth, self.depth * 2, kernel_size=2, stride=2, padding=0)
        self.conv2 = self.double_conv(self.depth * 2, self.depth * 2)
        
        self.down2 = nn.Conv1d(self.depth * 2, self.depth * 4, kernel_size=2, stride=2, padding=0)
        self.conv3 = self.double_conv(self.depth * 4, self.depth * 4)
        
        self.down3 = nn.Conv1d(self.depth * 4, self.depth * 8, kernel_size=2, stride=2, padding=0)
        self.conv4 = self.double_conv(self.depth * 8, self.depth * 8)
        
        self.up1 = self.make_up(self.depth * 8, self.depth * 4)
        self.conv5 = self.double_conv(self.depth * 4, self.depth * 4)
        
        self.up2 = self.make_up(self.depth * 4, self.depth * 2)
        self.conv6 = self.double_conv(self.depth * 2, self.depth * 2)
        
        self.up3 = self.make_up(self.depth * 2, self.depth)
        self.conv7 = self.double_conv(self.depth, self.depth)

        #last layer 
        self.classifier = nn.Conv1d(self.depth, self.out_channels, kernel_size=1)

    def double_conv(self, in_channels, depth):
        return nn.Sequential(
            nn.Conv1d(in_channels, depth, self.kernel_size, padding=2),
            nn.BatchNorm1d(depth),
            nn.ReLU(inplace=True),
            nn.Conv1d(depth, depth, self.kernel_size, padding=2, bias=False),
            nn.BatchNorm1d(depth ),
            nn.ReLU(inplace=True),
            #nn.Dropout(self.dropout_rate)
        )
    
    def make_up(self, in_channels, depth):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(in_channels, depth, kernel_size=3, padding=1),
            nn.BatchNorm1d(depth),
            nn.ReLU(inplace=True)
        )


    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_d=self.dropout(conv1)
        down1 = self.down1(conv1_d)
        down1= F.relu(down1)
        conv2 = self.conv2(down1)
        conv2_d=self.dropout(conv2)
        
        y = torch.add(down1 , conv2_d)
        down2 = self.down2(y)
        down2=F.relu(down2)
        conv3 = self.conv3(down2)
        conv3_d=self.dropout(conv3)
        

        y = torch.add(down2 , conv3_d)
        down3 = self.down3(y)
        down3=F.relu(down3)
        conv4 = self.conv4(down3)
        conv4_d=self.dropout(conv4)
        
        #PATCH_DIM/4
        up1 = self.up1(conv4_d)
        y = torch.add(conv3_d , up1)
        pre_conv5=self.conv5(y)
        pre_conv5_d=self.dropout(pre_conv5)
        conv5 = torch.add(pre_conv5_d , up1)
        
        #PATCH_DIM/2
        up2 = self.up2(conv5)
        y = torch.add(conv2 , up2)
        pre_conv6=self.conv6(y)
        pre_conv6_d=self.dropout(pre_conv6)
        conv6 = torch.add(pre_conv6_d , up2)
        
        #PATCH_DIM
        up3 = self.up3(conv6)
        y = torch.add(conv1 , up3)
        pre_conv7=self.conv7(y)
        pre_conv7_d=self.dropout(pre_conv7)
        conv7 = torch.add(pre_conv7_d , up3)
       

        output=self.classifier(conv7)
        softmax=F.softmax(output, dim=1)
        
        return softmax
    
    def dice_coef(self, y_true, y_pred, smooth=1e-8):
        intersection = torch.sum(y_true * y_pred, dim=2)
        return (2. * intersection + smooth) / (torch.sum(y_true, dim=2) + torch.sum(y_pred, dim=2) + smooth)

    def dice_loss(self, y_true, y_pred, weight):
        weight = weight.view(1, -1).to(y_true.device)
        weighted_dice=self.dice_coef(y_true, y_pred)*weight
        return 1 - weighted_dice.mean()