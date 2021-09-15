#from __future__ import print_function
"""
Updated log
time(m/d)   descripton    
08/08       
"""
import torch
import torchvision
from torch.autograd import Variable
import gan as mod
from torch.utils.data import Dataset, DataLoader
import data_loader as ds
from torchvision import transforms
import torch.nn as nn

import cv2
import numpy as np
import PIL as Image
import matplotlib.pyplot as plt
import math

import traceback
import os
from time import time
import argparse



TRAIN_DATA_NAME = './train.csv'
TEST_DATA_NAME = './test.csv'

CHECKPOINT_NAME = 'ckpt.pth.tar'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=8,
        help='Number of paired data to use in each update.')
    parser.add_argument("--batchgroups", type=int, default=1,
        help='Batches to accumulate before updating the weights.')
    parser.add_argument("--patchnum", type=int, default=1,
        help='Number of patches to extract from each image.')
    parser.add_argument("--patchsize", type=int, default=424,
        help='Size of the patches to use (in pixels).')
    parser.add_argument("--load", type=str, default='none', 
        help="Name of the file to load and continue training. \'none\' defaults to training from scratch.")
    parser.add_argument("--saveinterval", type=int, default=2500,
        help='Number of iterations between each save.')
    parser.add_argument("--trainfiles", type=str, default=TRAIN_DATA_NAME,
        help='Training file.')
    parser.add_argument("--scaledata", type=str, default='1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5',
        help='Different scales to use when training (comma-separated list).')
    parser.add_argument("--kaiten", type=int, default=180,
        help='Degrees to randomly rotate when training.')
    parser.add_argument("--epoch", type=int, default=5, 
        help='The training iteration.')
    parser.add_argument("--ckpt", type=int, default=2, 
        help='checkpoint.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
    parser.add_argument('--moduleName', default='model.t7', type=str,
        help='saved module name')
    parser.add_argument('--startepoch', default=0, type=int, metavar='N',
        help='manual epoch number (useful on restarts)')
    return parser



def main():
    args = vars(get_args().parse_args())
    
    print("=====The arguments======")
    for i in args:
        print(i, ":\t", args[i])
    print("========================\n\n")

    trainData = ds.Data_loader(TRAIN_DATA_NAME, transform=transforms.ToTensor())   
    trainLoader = DataLoader(dataset=trainData, batch_size=args['batchsize'],shuffle=True, drop_last=True)
    testData = ds.Data_loader(TEST_DATA_NAME, transform=transforms.ToTensor())
    testLoader = DataLoader(dataset=testData, batch_size=1)


    
    if(torch.cuda.is_available() == False):
        print("Using cpu to run it...")
        G = mod.Generator().to(DEVICE)
        D = mod.Discriminator().to(DEVICE)
    else:
        print("Using gpu to run it...")
        G = torch.nn.DataParallel(mod.Generator().cuda())
        D = torch.nn.DataParallel(mod.Discriminator().cuda())
    print("build model completed.")

    Gopt = torch.optim.Adam(params = G.parameters(),lr = 0.0002,betas = (0.5,0.999))
    Dopt = torch.optim.Adam(params = D.parameters(),lr = 0.0002,betas = (0.5,0.999))
    #loss_function = mod.OwnLossFunc().cuda()
    loss_fun = mod.L1Loss().cuda()
    #MSELoss = mean square error = L2 loss
    #According to papaer, color prediction network use L2 loss
    criterion = nn.BCELoss() 
    try:

        best_acc =0
        if args['resume'] != '':
            print('read module from ckpt')
            '''generator read'''
            if os.path.isfile('G' + args['resume']):
                if torch.cuda.is_available():
                    ckpt = torch.load('./G' + args['resume'])
                else:
                    ckpt = torch.load('./G' + args['resume'],map_location = 'cpu' )
                args['startepoch'] = ckpt['epoch']
                best_acc = ckpt['best_accuracy']
                G.load_state_dict(ckpt['state_dict'])
            '''D read'''
            if os.path.isfile('./D_' + args['resume']):
                if torch.cuda.is_available():
                    ckpt = torch.load('./D' + args['resume'])
                else:
                    ckpt = torch.load('./D' + args['resume'],map_location = 'cpu' )
                args['startepoch'] = ckpt['epoch']
                best_acc = ckpt['best_accuracy']
                D.load_state_dict(ckpt['state_dict'])

                print("load checkpoint '{}'(trained for {} epochs)".format(args['resume'],ckpt['epoch']))
            else:
                print('load failed: file not found.')

        #batch_x: gray data
        #batch_y: edge  data
        #batch_z: target  
        total_time = 0
        train_acc = 0
        real_label = 1
        fake_label = 0
        total_batch_num = (len(trainLoader))

        for epoch in range(args['startepoch'],args['epoch']):
            ''' d training but not sure input  data'''
            #print('epoch:' + str(epoch))
            gen_tloss = 0
            dis_tloss = 0
            tloss = 0
            start = time()
            lamb = 100      # for L2 loss
            print('epoch {} / {}'.format(epoch+1,args['epoch']) )

            for i,  (batch_y,batch_z) in enumerate(trainLoader):
                #print("round : %d"  %(i+1))
                #print('start to train discriminator...')
                #real train  
                D.zero_grad()
                d_real_out = D(batch_z)
                
                d_real_error = criterion(d_real_out,Variable(torch.ones(d_real_out.shape)).cuda())
                #d_real_error.backward()
                '''
                #fake gray train
                d_fake_data = G(batch_x)
                d_fake_out = D(d_fake_data.detach())
                d_fake_error = criterion(d_fake_out,Variable(torch.zeros(d_fake_out.shape)).cuda())
                d_fake_error.backward()
                '''
                #fake edge train
                de_fake_data = G(batch_y)
                de_fake_out = D(de_fake_data[3].detach())
                de_fake_error = criterion(de_fake_out,Variable(torch.zeros(de_fake_out.shape)).cuda())
                total_err = d_real_error + de_fake_error
                total_err.backward()
                #de_fake_error.backward()
                Dopt.step()

                #print('training discriminator done.')
                #print('start to train generator...')
                G.zero_grad()
    

                #edge
                eoutput = D(de_fake_data[3])
                eg_errorGan = criterion(eoutput,Variable(torch.ones(eoutput.shape)).cuda())
                eg_errorL1 = loss_fun(de_fake_data[3],batch_z.cuda())
                eg_error = eg_errorGan + lamb * eg_errorL1
                eg_error.backward()
                Gopt.step()
                gen_tloss += eg_errorL1
                dis_tloss += eg_errorGan
                tloss += eg_error
                #print('training generator done.')
                #print('round end.')


                progress = '[..................................................]'
                precent = math.ceil( (i+1) / total_batch_num * 50)
                
                cur = progress.replace('.','=',precent)
                cur = cur.replace('.','>')
                
                print(cur,'({}/{})'.format((i+1),total_batch_num),',loss:{}'.format(tloss / (i+1) / args['batchsize']),
                end='\r',flush = True)

            print(cur,'({}/{})'.format((i+1),total_batch_num),',loss:{}'.format(tloss / (i+1) / args['batchsize']),
                end='\n')
            total_time = total_time + time() - start
            print("Epoch %d.  Time %.1f sec." % (
            epoch+1,  time() - start))
            if(epoch % args['ckpt'] == 0):
                torch.save({'epoch': args['startepoch'] + epoch + 1,
                        'state_dict': G.state_dict(),
                        'best_accuracy': train_acc},'./G_' + CHECKPOINT_NAME)

                torch.save({'epoch': args['startepoch'] + epoch + 1,
                        'state_dict': D.state_dict(),
                        'best_accuracy': train_acc},'./D_' + CHECKPOINT_NAME)
                print('checkpoint reached')


        print('validating')
        test_loss = 0
        for i,  (batch_y,batch_z) in enumerate(testLoader):
            out = G(batch_y)
            loss_fun(out, batch_z.cuda())
            test_loss += loss_fun

        print('test loss: {}', test_loss // len(testLoader))

    except:
            traceback.print_exc()
            pass

    torch.save(G,'G'+args['moduleName'])
    torch.save(D,'D'+args['moduleName'])
    print('training compelete or being interrupted, save module, total use %.2f sec.' %(total_time))
    
if __name__=="__main__":
    main()

            


