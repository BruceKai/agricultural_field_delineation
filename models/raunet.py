import torch
import torch.nn as nn
from models.spatial_branch import InputConv
from torchvision import models
resnet=models.resnet50(pretrained=True)



class upblock(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        super().__init__()
        mid_channels=in_channels
        self.block=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
    def forward(self,x):
        return self.block(x)

class GatedAttention(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        output_channels = input_channels
        self.W_g = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(output_channels)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(output_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(output_channels, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(nn.Conv2d(input_channels*2,
                                    output_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True),
                            nn.BatchNorm2d(output_channels),
                            nn.ReLU(inplace=True))

    def forward(self,X_c,X_f):
        g1 = self.W_g(X_c)
        x1 = self.W_x(X_f)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        out = self.conv(torch.cat([X_c,X_f*psi],dim=1))
        return out,psi   

##  spatial encoder for high resolution imagries
class RAUNet(nn.Module):
    def __init__(self,in_channels,num_classes,bilinear=False):
        super(RAUNet,self).__init__()
        self.in_channels = in_channels
        filters = [64,256,512,1024,2048]
        self.bilinear = bilinear
     
        if self.in_channels ==3:
            self.conv1=resnet.conv1
        else:       
            self.conv1=nn.Conv2d(self.in_channels, 64, 
                                   kernel_size=(7, 7), stride=(2, 2), 
                                   padding=(3, 3), bias=False)               
        self.bn1=resnet.bn1
        self.relu=resnet.relu
        
        self.maxpool=resnet.maxpool       
        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4
        
        self.up_layer1 = upblock(filters[4],filters[3])
        self.ga1 = GatedAttention(filters[3])
        
        self.up_layer2 = upblock(filters[3],filters[2])
        self.ga2 = GatedAttention(filters[2])
        
        self.up_layer3 = upblock(filters[2],filters[1])
        self.ga3 = GatedAttention(filters[1])
        
        self.up_layer4 = upblock(filters[1],filters[0])
        self.ga4 = GatedAttention(filters[0])     
        
        self.up_layer5 = upblock(filters[0],filters[0]) 
        
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
        x = self.conv1(x)         # shape: (256,256) -> (128,128)
        x = self.bn1(x)
        # level0 = self.relu(self.bn1(level0))
        level1 = self.relu(x)     # shape: (64,128,128)
        level2 = self.maxpool(level1)    
        level2 = self.layer1(level2)   # shape: (256,64,64)
        level3 = self.layer2(level2)  # shape: (512,32,32)
        level4 = self.layer3(level3)  # shape: (1024,16,16),
        level5 = self.layer4(level4)  # shape: (2048,8,8)
        
        out = self.up_layer1(level5)
        out,attn1 = self.ga1(out,level4)
        out = self.up_layer2(out)
        out,attn2 = self.ga2(out,level3)
        out = self.up_layer3(out)
        out,attn3 = self.ga3(out,level2)
        out = self.up_layer4(out)
        out,attn4 = self.ga4(out,level1)
        out = self.up_layer5(out)
      
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
