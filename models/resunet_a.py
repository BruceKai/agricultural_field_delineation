""" ResUNet_a construction (ResUNet_a D7 with conditioned multi-task learning for agricultural field boudnary delineation)
 "Diakogiannis, F.I., Waldner, F., Caccetta, P., Wu, C., 2020. Resunet-a: a deep learning framework for semantic segmentation of remotely sensed data. ISPRS J. Photogramm. Remote Sens. 162, 94-114." 
"""

import torch 
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,dilation):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
            nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels,out_channels,3,
                   dilation=d,padding=d),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,out_channels,3,
                  dilation=d,padding=d),
                )
                for d in dilation
            ]
        )
       
    def forward(self,x):
        for conv in self.conv_list:
            x = x+conv(x)
        return x

def conv_block(in_channels,out_channels):
    
    conv = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels,1,
                            stride=1,padding=0),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                        )
    return conv
    

class encoder(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        filters = [32,64,128,256,512,1024,2048]
        self.conv1 = conv_block(in_channels,32)
        
        self.res_block1 = ResidualBlock(32,filters[0],[1,3,15,31])
        self.down_conv1 = nn.Conv2d(filters[0],filters[1],kernel_size=1,stride=2)
        
        self.res_block2 = ResidualBlock(filters[1],filters[1],[1,3,15,31])
        self.down_conv2 = nn.Conv2d(filters[1],filters[2],kernel_size=1,stride=2)
        
        self.res_block3 = ResidualBlock(filters[2],filters[2],[1,3,15])
        self.down_conv3 = nn.Conv2d(filters[2],filters[3],kernel_size=1,stride=2)
        
        self.res_block4 = ResidualBlock(filters[3],filters[3],[1,3,15])
        self.down_conv4 = nn.Conv2d(filters[3],filters[4],kernel_size=1,stride=2)  
        
        self.res_block5 = ResidualBlock(filters[4],filters[4],[1])
        self.down_conv5 = nn.Conv2d(filters[4],filters[5],kernel_size=1,stride=2)  
        
        self.res_block6 = ResidualBlock(filters[5],filters[5],[1])
        self.down_conv6 = nn.Conv2d(filters[5],filters[6],kernel_size=1,stride=2)  
        
        self.res_block7 = ResidualBlock(filters[6],filters[6],[1])
        
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(filters[6]*2,filters[6],1)
        
    def forward(self,x):
        x0 = self.conv1(x)
        x1 = self.res_block1(x0)
        x2 = self.res_block2(self.down_conv1(x1))
        x3 = self.res_block3(self.down_conv2(x2))
        x4 = self.res_block4(self.down_conv3(x3))
        x5 = self.res_block5(self.down_conv4(x4))
        x6 = self.res_block6(self.down_conv5(x5))
        x7 = self.res_block7(self.down_conv6(x6))
        x8 = F.interpolate(self.pool(x7),(x7.shape[-2],x7.shape[-1]),mode='bilinear')
        x7 = self.conv2(torch.cat([x7,x8],dim=1))
        return [x0,x1,x2,x3,x4,x5,x6,x7]

class PSPpooling(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
            nn.Sequential(
                nn.MaxPool2d(kernel_size=i,stride=i),
                nn.Conv2d(in_channels,in_channels//4,1),
                nn.BatchNorm2d(in_channels//4),
                )
                for i in [1,2,4,8]
            ]
        )
        self.conv = nn.Sequential(nn.Conv2d(in_channels*2,in_channels,1),
                                  nn.BatchNorm2d(in_channels))
       
    def forward(self,x):
        feat = []
        for conv in self.conv_list:
            feat.append(F.interpolate(conv(x),(x.shape[-2],x.shape[-1]),mode='bilinear'))
        feat.append(x)
        out = self.conv(torch.cat(feat,dim=1))
        return out

class ResUnet_a(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(ResUnet_a,self).__init__()
        filters = [32,64,128,256,512,1024,2048]
        self.num_classes = num_classes
        self.encoder = encoder(in_channels)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.combine1 = conv_block(filters[-1]+filters[-2],filters[-2])
        self.res_block1 = ResidualBlock(filters[-2],filters[-2],[1])

        self.combine2 = conv_block(filters[-2]+filters[-3],filters[-3])
        self.res_block2 = ResidualBlock(filters[-3],filters[-3],[1])       
        
        self.combine3 = conv_block(filters[-3]+filters[-4],filters[-4])
        self.res_block3 = ResidualBlock(filters[-4],filters[-4],[1,3,15])
        
        self.combine4 = conv_block(filters[-4]+filters[-5],filters[-5])
        self.res_block4 = ResidualBlock(filters[-5],filters[-5],[1,3,15])

        self.combine5 = conv_block(filters[-5]+filters[-6],filters[-6])
        self.res_block5 = ResidualBlock(filters[-6],filters[-6],[1,3,15,31])  
        
        self.combine6 = conv_block(filters[-6]+filters[-7],filters[-7])
        self.res_block6 = ResidualBlock(filters[-7],filters[-7],[1,3,15,31])
        
        self.combine7 = conv_block(filters[-7]+32,32)
        
        self.psp = PSPpooling(32)
        
        self.distance_block = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32,32,3,padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(inplace=True))
        self.semantic_block = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32,32,3,padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(inplace=True))   
        self.boundary_block = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(inplace=True))   
        self.distance_conv = nn.Conv2d(32,1,1)
        self.semantic_block = nn.Conv2d(32,num_classes,1)
        self.boundary_conv = nn.Conv2d(32,1,1)
       
    def forward(self,x):
        
        x0,x1,x2,x3,x4,x5,x6,x7 = self.encoder(x)
        
        x7 = self.upsample(x7)
        out = self.combine1(torch.cat([x6,x7],dim=1))
        out = self.res_block1(out)
        
        out = self.upsample(out)
        out = self.combine2(torch.cat([x5,out],dim=1))
        out = self.res_block2(out)
        
        out = self.upsample(out)
        out = self.combine3(torch.cat([x4,out],dim=1))
        out = self.res_block3(out)  
        
        out = self.upsample(out)
        out = self.combine4(torch.cat([x3,out],dim=1))
        out = self.res_block4(out)    
                
        out = self.upsample(out)
        out = self.combine5(torch.cat([x2,out],dim=1))
        out = self.res_block5(out)
                
        out = self.upsample(out)
        out = self.combine6(torch.cat([x1,out],dim=1))
        out = self.res_block6(out)
        
        out = self.combine7(torch.cat([x0,out],dim=1))
        
        distance_feat = self.distance_block(out)
        out = self.psp(out)
        distance = torch.sigmoid(self.distance_conv(distance_feat))
        
        boundary_feat = self.boundary_block(distance_feat+out)
        boundary = torch.sigmoid(self.boundary_conv(boundary_feat))
        
        extent = self.extent_block(distance_feat+boundary_feat+out)
        extent = self.extent_conv(extent)
        
        if self.num_classes == 1:
          extent = torch.sigmoid(extent)
        else:
          extent =  torch.softmax(extent,dim=1)
        
        prediction = {'extent':extent,
                     'boundary':boundary,
                     'distance':distance}
        
        return prediction
