import cv2
import numpy as np

def get_scretch(img):
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nimg = 255-img
    k = np.ones((3,3),np.uint8)
    nimg = cv2.erode(nimg,k)

    out = img + nimg
    
    #out[out > 10] += 80
    out = np.clip(out,0,255)
    #out = -out
    #_,out = cv2.threshold(out,40,255,cv2.THRESH_BINARY)
    #out = 255 -out
    return out


img = cv2.imread('test.jpg')
out = get_scretch(img)
cv2.imwrite('t_test.jpg',out)
cv2.imshow('rst',out)
cv2.waitKey(0)