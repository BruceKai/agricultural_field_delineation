""" SEANet construction (ResUNet_a D7 with conditioned multi-task learning for agricultural field boudnary delineation)
 "Mengmeng, Li., Jiang, Long., Alfred, Stein., Xiaoqin Wang., 2023. 
 Using a semantic edge-aware multi-task neural network to delineate agricultural parcels from remote sensing images. 200, 24-40. https://doi.org/10.1016/j.isprsjprs.2023.04.019" 
"""


import torch 
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models as torch_models
vgg16 = torch_models.vgg16(pretrained=True)

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DecoderBlock,self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels,3,
                            stride=1,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels,out_channels,3,
                            stride=1,padding=1),
                        nn.ReLU(inplace=True))
    def forward(self,x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class PEoI(nn.Module):
    def __init__(self,):
        super(PEoI,self).__init__()
        filters = [64,128,256,512,512]
        self.conv_list = nn.ModuleList([nn.Sequential(
                        nn.Conv2d(filters[i],21,3,
                            stride=1,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(21,1,3,
                            stride=1,padding=1),
                        nn.ReLU(inplace=True)) for i in range(5)])
        self.conv = nn.Conv2d(5,1,1)
    def forward(self,x):
        feat = []
        for i in range(5):
            tmp_feat = self.conv_list[i](x[i])
            if i >0:
                tmp_feat = F.interpolate(tmp_feat,
                                         (x[0].shape[-2],x[0].shape[-1]),
                                         mode='bilinear')
            feat.append(tmp_feat)
        
        feat = torch.cat(feat,dim=1)
        edge = torch.sigmoid(self.conv(feat))
        return edge        

class ASPPblock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,dilation,is_pool):
        super(ASPPblock,self).__init__()
        padding = 0 if kernel_size==1 else dilation
        if is_pool:
            self.aspp = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0),
                nn.ReLU(inplace=True) 
                )
        else:
            self.aspp = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                          dilation=dilation,padding=padding
                          ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) 
                )
    def forward(self,x):
        x = self.aspp(x)
        return x

class ASPP(nn.Module):
    def  __init__(self,in_channels,out_channels):
        super(ASPP,self).__init__()
        dilation = [1,2,4]
        self.aspp1 = ASPPblock(in_channels,out_channels,1,1,False)
        self.aspp2 = ASPPblock(in_channels,out_channels,3,dilation[0],False)
        self.aspp3 = ASPPblock(in_channels,out_channels,3,dilation[1],False)
        self.aspp4 = ASPPblock(in_channels,out_channels,3,dilation[2],False)
        self.aspp5 = ASPPblock(in_channels,out_channels,1,dilation[0],True)
        
        self.conv = nn.Sequential(nn.Conv2d(out_channels*5, out_channels, 1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
    
    def forward(self,x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp5(x)
        x5 = F.interpolate(x5,(x.shape[-2],x.shape[-1]),mode='bilinear')
        out = self.conv(torch.cat([x1,x2,x3,x4,x5],dim=1))
        return out



class SEANet(nn.Module):
    def __init__(self,in_channels,**kwargs):
        super(SEANet, self).__init__()
        filters = [64,128,256,512]
        
        if in_channels == 3:
            self.conv_block1 = vgg16.features[:4]
        else:
            self.conv_block1 = nn.Sequential(
                        nn.Conv2d(in_channels,64,3,
                        stride=1,padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64,64,3,
                        stride=1,padding=1),
                        nn.ReLU(inplace=True),
                        )
        self.conv_block2 = vgg16.features[4:9]
        self.conv_block3 = vgg16.features[9:16]
        self.conv_block4 = vgg16.features[16:23]
        self.conv_block5 = vgg16.features[23:]
        
        self.peoi = PEoI()
        self.aspp1 = ASPP(512,256)
        self.decoder_b1 = DecoderBlock(512+256,256)
        self.decoder_b2 = DecoderBlock(256+256,256)
        self.decoder_b3 = DecoderBlock(256+128,128)
        
        self.aspp2 = ASPP(128,64)
        self.extent_conv = nn.Conv2d(64,1,1)
        self.distance_conv = nn.Conv2d(64,1,1)


        
    def forward(self,x):
        
        level1 = self.conv_block1(x)
        level2 = self.conv_block2(level1)
        level3 = self.conv_block3(level2)
        level4 = self.conv_block4(level3)
        level5 = self.conv_block5(level4)
        
        boundary = self.peoi([level1,level2,level3,level4,level5])
        feat = self.aspp1(level4)
        
        feat = self.decoder_b1(torch.cat([feat,level4],dim=1))
        feat = self.decoder_b2(torch.cat([feat,level3],dim=1))
        feat = self.decoder_b3(torch.cat([feat,level2],dim=1))
        
        feat = self.aspp2(feat)
        
        extent = torch.sigmoid(self.extent_conv(feat))
        distance = torch.sigmoid(self.distance_conv(feat))
        prediction = {'extent':extent,
                      'boundary':boundary,
                      'distance':distance}


        return prediction

