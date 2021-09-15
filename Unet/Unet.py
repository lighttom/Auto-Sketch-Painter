import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Unet, encoder -> VGG 16
# decoder -> decoder


class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model,self).__init__()

        # vgg encoder:
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),

            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
        )

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv2 = nn.Sequential(
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),      
        )

        
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )


        self.conv3 = nn.Sequential(
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),            
        )

        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv4 = nn.Sequential(
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(),
        )

        self.pool4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.psss_layer = nn.Sequential(
            
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features =1024),
            nn.ReLU(),

            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features =1024),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=1024,padding = 1,out_channels=512,kernel_size=4,stride=2)
        )

        self.up_conv4 = nn.Sequential(
            
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features =512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features =512),
            nn.ReLU(),


        )
        self.up_sample_4 = nn.ConvTranspose2d(in_channels=512,out_channels=256,padding=1,kernel_size=4,stride=2)
        self.out_4 = nn.Conv2d(in_channels=512,out_channels=3,kernel_size = 1, stride = 1)


        self.up_conv3 = nn.Sequential(
            
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features =256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features =256),
            nn.ReLU(),

            
        )
        self.up_sample_3 = nn.ConvTranspose2d(in_channels=256,out_channels=128,padding=1,kernel_size=4,stride=2)

        self.out_3 = nn.Conv2d(in_channels=256,out_channels=3,kernel_size = 1, stride = 1)

        self.up_conv2 = nn.Sequential(
            
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features =128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128,out_channels=128,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features =128),
            nn.ReLU(),

            
        )
        self.up_sample_2 = nn.ConvTranspose2d(in_channels=128,out_channels=64,padding=1,kernel_size=4,stride=2)

        self.out_2 = nn.Conv2d(in_channels=128,out_channels=3,kernel_size = 1, stride = 1)

        self.up_conv1 = nn.Sequential(
            
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features =64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size = 3, stride = 1,padding = 1,padding_mode='replicate'),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=3,kernel_size = 1, stride = 1)
        )

        



    def forward(self,x):

        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool3(conv4)

        p_layer = self.psss_layer(pool4)

        up_conv4 = self.up_conv4(torch.cat((p_layer,conv4),1))
        out_4 = self.out_4(up_conv4)
        up_s4 = self.up_sample_4(up_conv4)
        up_conv3 = self.up_conv3(torch.cat((up_s4,conv3),1))
        out_3 = self.out_3(up_conv3)
        up_s3 = self.up_sample_3(up_conv3)
        up_conv2 = self.up_conv2(torch.cat((up_s3,conv2),1))
        out_2 = self.out_2(up_conv2)
        up_s2 = self.up_sample_2(up_conv2)
        out = self.up_conv1(torch.cat((up_s2,conv1),1))



        return out_4, out_3, out_2, out
        #return out

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x1,x2,x3,x4,y):
        
        b,c,w,h = y.shape
        w//=2
        h//=2
        y2 = F.interpolate(y,(w,h))
        
        h //= 2
        w //= 2

        y3 = F.interpolate(y,(w,h))

        h //= 2
        w //= 2

        y4 = F.interpolate(y,(w,h))
        l1 = torch.nn.MSELoss()(x1,y)
        l2 = torch.nn.MSELoss()(x2,y2)
        l3 = torch.nn.MSELoss()(x3,y3)
        l4 = torch.nn.MSELoss()(x4,y4)

        total_loss =  l1 + l2 + l3 + l4
        return total_loss

class discriminator(nn.Module):
    def __init__(self,num_class):
        super(discriminator,self).__init__()
        #downConv1
        self.downConv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size = 5,stride = 2,padding = 2),
            nn.BatchNorm2d(num_features = 64),
            nn.LeakyReLU()
        )

        #downConv2
        self.downConv2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size = 5,stride = 2,padding = 2),
            nn.BatchNorm2d(num_features = 128),
            nn.LeakyReLU()
        )        

        #downConv3
        self.downConv3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size = 5,stride = 2,padding = 2),
            nn.BatchNorm2d(num_features = 256),
            nn.LeakyReLU()
        )

        #downConv4
        self.downConv4 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size = 5,stride = 2,padding = 2),
            nn.BatchNorm2d(num_features = 512),
            nn.LeakyReLU()
        )

        #downConv5
        self.downConv5 = nn.Sequential(
            nn.Conv2d(512,1,kernel_size = 5,stride = 1,padding = 1),
            nn.Sigmoid()
        )

        self.downConv5_2 = nn.Sequential(
            nn.Conv2d(512,1,kernel_size = 5,stride = 1,padding =1),
            nn.Linear(512,num_class),
            nn.Softmax()
            )

    def forward(self,x):
        out = self.downConv1(x)
        out = self.downConv2(out)
        out = self.downConv3(out)
        out = self.downConv4(out)
        fake = self.downConv5 (out)
        #label = self.downConv5_2(out)
        return  fake
        """
        sigmoid funciton is used to make sure that 
        the return value is within the range of 0 to 1.
        """



        