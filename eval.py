import torch.nn
import numpy as np
import cv2
from PIL import Image
import os

MODEL_NAME = './Gmodel.t7'

def resize(img):
    MAX_X = 1120
    MAX_Y = 960
    y,x,c = img.shape
    print(img.shape)
    new_y = y - y % 32
    new_x = x - x % 32 
    
    if new_x >= MAX_X:
        new_x = MAX_X
        new_y = int(new_y * (new_x / MAX_X ))
        new_y = new_y - new_y % 32
    if new_y > MAX_Y:
        new_y = MAX_Y
        new_x = int(new_x * (new_y / MAX_Y ))
        new_x = new_x - new_x % 32

    out = cv2.resize(img,(new_x,new_y),interpolation=cv2.INTER_CUBIC)
    print(out.shape)
    return out,y,x 
def painting(model,img):

    # img = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
    img,oy,ox = resize(img)

    y_size,x_size,c = img.shape
    if c == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img,0)
    img = np.expand_dims(img,-1)
    
    x = torch.tensor(img).to("cuda",dtype=torch.float32)
    x = x.permute(0,3,2,1)
    print(x.shape)
    
    _,_,_, out = model(x)
    out = torch.squeeze(out)
    out = out.permute(1,2,0)
    out = out.cpu()
    out = out.detach()
    out = out.numpy()
    # if y_size != 256 or x_size != 256:
    #     out = cv2.resize(out,(x_size,y_size),interpolation=cv2.INTER_CUBIC)
    out = out[::-1,:,:]
    out = out * 255
    out = np.clip(out,0,255)
    out = out.astype(np.uint8)
    po = Image.fromarray(out)
    po = po.rotate(270,expand=1)


    out = np.asarray(po)
    
    out = cv2.resize(out,(ox,oy),interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('out',out)
    #cv2.waitKey(0)
    
    return out
if __name__ == '__main__':


    model = torch.load(MODEL_NAME)
    
    use_gpu = True
    if use_gpu:
        model = model.cuda()
    

    INDIR = './test_img'
    OUTDIR = './test_img_out'
    IMG_NAME = './9_test.jpg'
    # img = cv2.imread(IMG_NAME)
    # out = painting(img)
    # cv2.imwrite('.'+IMG_NAME.split('.')[1] + "_out.jpg",out)
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    for img_name in os.listdir(INDIR):
        img_path = os.path.join(INDIR,img_name)
        img = cv2.imread(img_path)

        out = painting(model,img)
        out_path = os.path.join(OUTDIR,img_name.split('.')[0]+'_out.png')
        cv2.imwrite(out_path,out)



