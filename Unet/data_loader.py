import csv
import torchvision.transforms
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import cv2
import numpy as np

def default_loader(path,mode='rgb'):
    img = cv2.imread(path)
    
    if mode == 'rgb':
        return img
    elif mode == 'lab':
        pimg = Image.fromarray(img)
        pimg.convert('LAB')
        img = np.asarray(pimg)

    return img

class Data_loader(Dataset):
    
    def __init__(self,datapath,transform = None,loader = default_loader):
        self.input_x,self.input_y = [], []
        data_csv = pd.read_csv(datapath)
        for val in data_csv.values:

            self.input_x.append(val[1])
            self.input_y.append(val[2])
        
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        train_x = self.loader(self.input_x[index])
        train_y = self.loader(self.input_y[index])
        if train_x.shape[2] == 3:
            train_x = cv2.cvtColor(train_x,cv2.COLOR_BGR2GRAY)

        if self.transform != None:
            train_x = self.transform(train_x)
            train_y = self.transform(train_y)

        return train_x, train_y
    
    def __len__(self):
        return len(self.input_x)
