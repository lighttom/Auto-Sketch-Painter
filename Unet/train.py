import Unet
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
from time import time
import datetime
import math

import data_loader

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8,
    help='set training batch_size, default 8')
    parser.add_argument("--epoch", type=int, default=1000,
    help='set training epoch, default 1000')
    parser.add_argument("--save_interval", type=int, default=-1,
    help='set save model interval, default = epoch // 100')
    parser.add_argument('--load', type=str, default=None,
    help='if continue, load model')
    parser.add_argument("--use_gpu",type=bool, default=True,
    help='set use gpu or not, default true')

    return vars(parser.parse_args())

def main():
    args = get_argument()
    

    # load dataset
    train_data = data_loader.Data_loader(datapath = './train.csv', transform = transforms.ToTensor())
    trainLoader = DataLoader(train_data,batch_size=args['batch_size'],shuffle=True,drop_last=True)
    test_data = data_loader.Data_loader(datapath = './test.csv', transform = transforms.ToTensor())
    testLoader = DataLoader(train_data,batch_size=1)

    # load model
    print('load model...')
    if args['use_gpu'] == True:
        DEVICE = 'cuda'
        print('use gpu')

        if torch.cuda.is_available() == True:
            model = Unet.Model().cuda()
        else:
            DEVICE = 'cpu'
            print('load gpu error, use cpu')
            model = Unet.Model().to(DEVICE)
    else:
        DEVICE = 'cpu'
        print('use cpu')
        model = Unet.Model().to(DEVICE)
    
    print('done')
    print('curent device: ',DEVICE)
    # setting optimzer
    
    opt = torch.optim.Adam(params = model.parameters(), lr = 0.001, betas=(0.5,0.999))
    loss_func = Unet.Loss().to(DEVICE)    
    
    all_start = time()
    one_epoch_time = 0
    total_batch_num = (len(trainLoader) // args['batch_size'])
    try:
        for i in range(args['epoch']):
            epo_start = time()
            total_loss = 0
            print('epoch {} / {}'.format(i+1,args['epoch']))
            

            for j,(batch_x,batch_y) in enumerate(trainLoader):
                bat_start = time()
                batch_x, batch_y = Variable(batch_x).to(DEVICE), Variable(batch_y).to(DEVICE)
                
                
                y4, y3, y2, y = model(batch_x)

                loss = loss_func(y, y2, y3, y4, batch_y)
                
                total_loss += loss.item()

                opt.zero_grad()
                loss.backward()
                opt.step()

                one_batch_time = time() - bat_start
                eta = one_batch_time * (total_batch_num - (j+1))
                progress = '[..................................................]'
                precent = math.ceil( (j+1) / total_batch_num * 50)
                cur = progress.replace('.','=',precent)
            
                
                print(cur,'({}/{})'.format((j+1),total_batch_num),',loss:{}'.format(total_loss / (j+1) / args['batch_size']),
                ',ETA: ',str(datetime.timedelta(eta)),end='\r')

            print(cur,'({}/{})'.format((j+1),total_batch_num),',loss:{}'.format(total_loss / (j+1) / args['batch_size']),
            ',ETA: ',str(datetime.timedelta(eta)),end='\n')
            
            print('epoch: %d, loss:%.3f, time: %.3f' %(
                i + 1, total_loss / len(trainLoader), time() - epo_start))
            
            if (i + 1) % args['save_interval'] == 0:
                torch.save({'epoch': args['epoch'] + 1,
                        'state_dict': model.state_dict(),
                        'loss': total_loss / len(trainLoader)},'./weights/model_{}.pth.tar'.format(args['epoch']+1))
                print('save state dict at epoch {}'.format(i+1))
        print('done')
        print('cost {} seconds'.format(time() - all_start))
        torch.save(model,'./weights/model_{}.7t'.format(args['epoch']+1))
    except KeyboardInterrupt:
        print('Interrupted!. save current model')
        torch.save(model,'./weights/model_{}_interrputed.7t'.format(args['epoch']+1))


if __name__ == '__main__':
    main()







    