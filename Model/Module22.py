import numpy as np
import random
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init

# Initialization of network parameters
def weights_init(m, leak_value):
    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.weight is not None:

            init.xavier_normal_(m.weight, leak_value)

        if m.bias is not None:
            init.constant_(m.bias, 0.0)

    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_normal_(m.weight,leak_value )
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

#create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self,D1):
        super(SiameseNetwork, self).__init__()
        self.D1 = D1
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Conv2d(1, self.D1, kernel_size=3,stride=1,padding=1)
        self.a1 = nn.LeakyReLU(0.5)
        self.cnn2 = nn.Conv2d(self.D1, 2*self.D1, kernel_size=3,stride=1,padding=1)
        self.a2 = nn.LeakyReLU(0.5)
        self.cnn3 = nn.Conv2d(2*self.D1, 2*self.D1, kernel_size=3,stride=1,padding=1)
        self.a3 = nn.LeakyReLU(0.5)
        
        self.cnn4 = nn.Conv2d(2*self.D1, 4*self.D1, kernel_size=3,stride=1,padding=1)
        self.a4 = nn.LeakyReLU(0.5)
        
        self.cnn5 = nn.Conv2d(4*self.D1, 4*self.D1, kernel_size=3,stride=1,padding=1)
        self.a5 = nn.LeakyReLU(0.5)
        self.cnn6 = nn.Conv2d(4*self.D1, 2*self.D1, kernel_size=3,stride=1,padding=1)
        self.a6 = nn.LeakyReLU(0.5)
        self.cnn7 = nn.Conv2d(2*self.D1, self.D1, kernel_size=3,stride=1,padding=1)
        self.a7 = nn.LeakyReLU(0.5)
        
        self.cnn8 = nn.Conv2d(self.D1, 1, kernel_size=3,stride=1,padding=1)
        
        self.cnnXX1 = nn.Conv2d(1, self.D1, kernel_size=3,stride=1,padding=1)
        self.axx1 = nn.LeakyReLU(0.5)

        self.cnnXX2 = nn.Conv2d(1, 2*self.D1, kernel_size=3,stride=1,padding=1)
        self.axx2 = nn.LeakyReLU(0.5)
        
        self.cnnXX3 = nn.Conv2d(1, 2*self.D1, kernel_size=3,stride=1,padding=1)
        self.axx3 = nn.LeakyReLU(0.5)
        
        self.cnnXX4 = nn.Conv2d(1, 4*self.D1, kernel_size=3,stride=1,padding=1)
        self.axx4 = nn.LeakyReLU(0.5)
       
        
        self.cnnXX5 = nn.Conv2d(1, 4*self.D1, kernel_size=3,stride=1,padding=1)
        self.axx5 = nn.LeakyReLU(0.5)
        
        self.cnnXX6 = nn.Conv2d(1, 2*self.D1, kernel_size=3,stride=1,padding=1)
        self.axx6 = nn.LeakyReLU(0.5) 
        
        self.cnnXX7 = nn.Conv2d(1, self.D1, kernel_size=3,stride=1,padding=1)
        self.axx7 = nn.LeakyReLU(0.5)
        self.cnnXX = nn.Conv2d(1, self.D1, kernel_size=3,stride=1,padding=1)
        self.axx = nn.LeakyReLU(0.5)
        
        
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        
        xx1 =  self.axx1(self.cnnXX1(x))
        xx2 =  self.axx2(self.cnnXX2(x))
        xx3 =  self.axx3(self.cnnXX3(x))
        xx4 =  self.axx4(self.cnnXX4(x))
        xx5 =  self.axx5(self.cnnXX5(x))
        xx6 =  self.axx6(self.cnnXX6(x))
        xx7 =  self.axx7(self.cnnXX7(x))
        xx =   self.cnnXX(x)
        
        output1 = self.a1(self.cnn1(x))
        output1 = output1 + xx1
        
        output2 = self.a2(self.cnn2(output1))
        #output2 = torch.cat((output1,output2),dim=1)
        output2 = output2 + xx2
       
        output3 = self.a3(self.cnn3(output2))
        #output3 = torch.cat((output2,output3),dim=1)
        output3 = output3 + xx3
        
        output4 = self.a4(self.cnn4(output3))
        output4 = output4 + xx4 
        
        output5 = self.a5(self.cnn5(output4))
        output5 = output5 + xx5

        output6 = self.a6(self.cnn6(output5))
        #output2 = torch.cat((output1,output2),dim=1)
        output6 = output6 + xx6 
       
        output7 = self.a7(self.cnn7(output6))
        #output3 = torch.cat((output2,output3),dim=1)
        output7 = output7 + xx7
       
        output8 = self.cnn8(output7)
        output8 =   output8 + xx + x
                            
        return output4, output8

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1,o1 = self.forward_once(input1)
        output2,o2 = self.forward_once(input2)

        return output1, output2,o1,o2
    
