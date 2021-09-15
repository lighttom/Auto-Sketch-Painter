import pandas as pd
import cv2
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt


def default_loader(imgPath):
    return Image.open(imgPath)

class MyDataSet(Dataset):
    
    def __init__(self, file, transform=None, loader=default_loader):
        
        inList, outList = [], []
        csvDF = pd.read_csv(file)
        for i in csvDF.values:
            inList.append(i[0])
            outList.append(i[1])
        self.inList = inList
        self.outList = outList
        self.transform = transform
        self.loader = loader
        
    def __getitem__(self, index):
        inPath = self.inList[index]
        outPath = self.outList[index]
        inImg = self.loader(inPath)
        outImg = self.loader(outPath)

        if(self.transform is not None):
            inImg = self.transform(inImg)
            outImg = self.transform(outImg)
        
        #show image
        # to_pil = torchvision.transforms.ToPILImage()
        #ins = to_pil(inImg)
        # plt.imshow(ins)
        #plt.show()
        
        return inImg, outImg

    def __len__(self):
        return len(self.inList)


"""
>>>it returns numpy array<<<

def np_get_data(trainName, testName):
    
    x_trainList, x_testList = [], []
    y_trainList, y_testList = [], []

    trainDF = pd.read_csv(trainName, sep=',', header=None)
    testDF = pd.read_csv(testName, sep=',', header=None)
    
    for i in trainDF.values:
        x_trainImg = cv2.imread(i[0])
        y_trainImg = cv2.imread(i[1])

        x_trainList.append(x_trainImg)
        y_trainList.append(y_trainImg)
    
    for i in testDF.values:
        x_testImg = cv2.imread(i[0])
        y_testImg = cv2.imread(i[1])

        x_testList.append(x_testImg)
        y_testList.append(y_testImg)


    x_train = np.array(x_trainList)
    x_test = np.array(x_testList)
    y_train = np.array(y_trainList)
    y_test = np.array(y_testList)

    return x_train, y_train, x_test, y_test
"""

