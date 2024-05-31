import numpy as np
import math
import cv2  


def gauss_func(x, y, sigma_x, sigma_y):
    coeff =  1 / (2 * math.pi * sigma_x * sigma_y) 
    val = (((x ** 2)/(sigma_x ** 2)) + ((y ** 2)/(sigma_y ** 2))) / 2
    value = math.exp(-(val) )
    value = value * coeff
    return value

def gaussian_kernel(sigma_x, sigma_y):
    k_row = int(5*sigma_x)
    k_col = int(5*sigma_y)

    if k_row%2==0:
        k_row=k_row+1       
    if k_col%2==0:
        k_col=k_col+1 
        
    gauss_kernel = np.zeros((k_row,k_col), dtype=np.float32)

    for i in range(0, k_row):
        for j in range(0, k_col):
            gauss_kernel[i,j] = gauss_func(i, j, sigma_x, sigma_y)
            

    return gauss_kernel

def convolution(kernel, image, c_x = 1, c_y = 1):
    w = kernel.shape[0]//2
    h = kernel.shape[1]//2
    
    pad_bottom = kernel.shape[0] - c_x -1
    pad_right = kernel.shape[1] - c_y -1
    
    img_bordered = cv2.copyMakeBorder(src=image, top=c_x, bottom=pad_bottom, left=c_y, right=pad_right,  borderType=cv2.BORDER_CONSTANT)
    cv2.imshow("Input Image Bordered", img_bordered)

    out = np.zeros((image.shape[0]+2*w, image.shape[1]+2*h))

    for i in range(c_x, img_bordered.shape[0] - pad_bottom - 1):
        for j in range(c_x, img_bordered.shape[1] - pad_right - 1):
            sum = 0
            for x in range(-w, w + 1):
                for y in range(-h, h + 1):
                    sum += kernel[x + w, y + h] * img_bordered[i - x, j - y]
            out[i, j] = sum    

    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    cv2.imshow("Output Image", out) 
         
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return out
