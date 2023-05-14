""" implementation of UNet for agricultural field boundary delineation
  @Author: Zhiwen.Cai  
  @Date: 2022-07-18 17:07:35  
  @Last Modified by:  Zhiwen.Cai  
  @Last Modified time: 2022-07-18 17:07:35  
"""
import torch 
import torch.nn as nn


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv,self).__init__()
        mid_channels=32
        self.out_channels = out_channels
        self.conv = nn.Sequential(nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
                                nn.BatchNorm2d(mid_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(mid_channels, out_channels, kernel_size=1)
                                ) 
    def forward(self, x):
        # x = self.conv(x)
        if self.out_channels > 1 :
            out = torch.softmax(self.conv(x),dim=1)

        else:
            out = torch.sigmoid(self.conv(x))

        return out


def conv_block(in_channels,out_channels,kernel_size=3):

    conv = nn.Sequential(
                        nn.Conv2d(in_channels,out_channels,kernel_size,
                        stride=1,padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels,out_channels,kernel_size,
                        stride=1,padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        )
    return conv                        

class encoder(nn.Module):
    def __init__(self,filters):
        super(encoder,self).__init__()
        # filters = [data_channels,64,128,256,512,1024]
        self.layer1 = conv_block(filters[0],filters[1])

        self.layer2 = nn.Sequential(
                                nn.MaxPool2d(2),
                                conv_block(filters[1],filters[2]),
                                )
        self.layer3 = nn.Sequential(
                                nn.MaxPool2d(2),
                                conv_block(filters[2],filters[3]),
                                )
        self.layer4 = nn.Sequential(
                                nn.MaxPool2d(2),
                                conv_block(filters[3],filters[4])      
                                )
        self.layer5 = nn.Sequential(
                                nn.MaxPool2d(2),
                                conv_block(filters[4],filters[5])      
                                )
    def forward(self,x):

        level1 = self.layer1(x)
        level2 = self.layer2(level1)
        level3 = self.layer3(level2)
        level4 = self.layer4(level3)
        level5 = self.layer5(level4)

        return level1,level2,level3,level4,level5

class up_conv(nn.Module):
    def __init__(self,
                  in_channels,
                  out_channels,
                  kernel_size=3,
                  mode='upsample'):
        super(up_conv,self).__init__()
        if mode == 'upsample':
            self.up_sample = nn.Sequential(
                                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(in_channels,in_channels//2,kernel_size=3,padding=1),
                                        nn.BatchNorm2d(in_channels//2),
                                        nn.ReLU(inplace=True),
                                        )                 
        else:
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = conv_block(in_channels,out_channels,kernel_size)
    def forward(self,coarse,fine):
        x = self.up_sample(coarse)
        x = self.conv(torch.cat([x,fine],dim=1))        
        return x

class decoder(nn.Module):
    def __init__(self,filters):
        super(decoder,self).__init__()
        # filters = [1024,512,256,128,64]
        self.layer1 = up_conv(filters[0],filters[1])
        self.layer2 = up_conv(filters[1],filters[2])                             
        self.layer3 = up_conv(filters[2],filters[3])                                  
        self.layer4 = up_conv(filters[3],filters[4])                            
                                    
    def forward(self,encoded_features):
        # level1 features shape (64,256,256)
        level1,level2,level3,level4,level5 = encoded_features
        x = self.layer1(level5,level4)
        x = self.layer2(x,level3)
        x = self.layer3(x,level2)
        out = self.layer4(x,level1)
        
        return out

class UNet(nn.Module):
    def __init__(self,in_channels,num_classes,**kwargs):
        super(UNet, self).__init__()
        filters = [in_channels,64,128,256,512,1024]
        self.num_classes = num_classes
        self.encoder = encoder(filters)
        filters_flip = filters[1:]
        filters_flip.reverse()     
        self.decoder = decoder(filters_flip)
        self.distance_conv = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32,32,3,padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32,1,1))
        self.extent_conv = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32,32,3,padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32,num_classes,1))   
        self.boundary_conv = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                                nn.BatchNorm2d(32),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32,1,1))   

        
    def forward(self,x):
        
        features = self.encoder(x)
        out = self.decoder(features)
        
        distance = torch.sigmoid(self.distance_conv(out))
        boundary = torch.sigmoid(self.boundary_conv(out))
        extent = self.extent_conv(out)
        if self.num_classes == 1:
          extent = torch.sigmoid(extent)
        else:
          extent =  torch.softmax(extent,dim=1)
        
        prediction = {'extent':extent,
                     'boundary':boundary,
                     'distance':distance}
        return prediction
