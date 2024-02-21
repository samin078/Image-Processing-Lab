import cv2 as cv
import numpy as np
import math

PI = 3.14159265

sigma_x = float(input("Enter sigma_x: "))
sigma_y = float(input("Enter sigma_y: "))

k_row = int(5*sigma_x)
k_col = int(5*sigma_y)

c_x = int(input("Enter center x: "))
c_y = int(input("Enter center y: "))

if k_row%2==0:
    k_row=k_row+1       #for center symmetric
if k_col%2==0:
    k_col=k_col+1       #for center symmetric

# gaussian function
def gauss_func(x, y):
    coeff =  1 / (2 * PI * sigma_x * sigma_y) 
    val = (((x ** 2)/(sigma_x ** 2)) + ((y ** 2)/(sigma_y ** 2))) / 2
    value = math.exp(-(val) )
    value = value * coeff
    return value

def gaussian_kernel():
    gauss_kernel = np.zeros((k_row,k_col), dtype=np.float32)

    for i in range(0, k_row):
        for j in range(0, k_col):
            gauss_kernel[i,j] = gauss_func(i,j)
            
    # print(gauss_kernel)
    return gauss_kernel


# define mean kernel 
def mean_kernel():
    mean_coeff = k_row * k_col
    mean_kernel = (1 / mean_coeff) * np.ones((k_row,k_col), dtype=np.uint8)

    # print(mean_kernel)
    return mean_kernel

# define Laplacian kernel 
def laplace_kernel_3():
    laplacian_kernel_3 = np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=np.float32)
    # print(laplacian_kernel_3)
    return laplacian_kernel_3

def laplace_kernel_5():
    laplacian_kernel_5 = np.array([ [0, 0,  1,  0, 0],
                                [0, 1,  2,  1, 0],
                                [1, 2, -16, 2, 1],
                                [0, 1,  2,  1, 0],
                                [0, 0,  1,  0, 0]], dtype=np.float32)
    # print(laplacian_kernel_5)
    return laplacian_kernel_5


def h_sobel_kernel():
    a = np.array([[1],
                  [2],
                  [1]], dtype=np.float32)

    b = np.array([1, 0, -1], dtype=np.float32)

    h_sobel_kernel = a * b
    return h_sobel_kernel

def v_sobel_kernel():
    c = np.array([[1],
                  [0],
                  [-1]], dtype=np.float32)

    d = np.array([1, 2, 1], dtype=np.float32)

    v_sobel_kernel = c * d
    return v_sobel_kernel
    
# kernel = gaussian_kernel()
# print(kernel)

def color_image():
    img_color = cv.imread('Lena.jpg',cv.IMREAD_COLOR)
    (blue ,green , red) = cv.split(img_color)
    # cv.imshow('color image',img_color) 
    # cv.imshow('blue image', blue) 
    # cv.imshow('green image', green) 
    # cv.imshow('red image', red) 

    hsv_image = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    (blue_hsv , green_hsv , red_hsv) = cv.split(hsv_image)
    # cv.imshow('hsv image', hsv_image) 
    # cv.imshow('blue hsv', blue_hsv) 
    return img_color

def gray_image():
    img_grayscale = cv.imread("Lena.jpg", cv.IMREAD_GRAYSCALE)
    # cv.imshow("Lena Grayscale", img_grayscale)
    return img_grayscale

img_type = 0

def img_choice(kernel):
    print ("""
    1.Grayscale Image
    2.RGB Image
    3.Back
    """)
    inp=input("Select Image: ") 
    if inp=="1": 
        print("\n Grayscale Image") 
        image = gray_image()
        cv.imshow('Input Image', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        out = convulation(kernel, image, c_x, c_y)
        return image
    elif inp=="2":
        print("\n Color Image") 
        image = color_image()
        cv.imshow('Input Image', image)
        (blue ,green , red) = cv.split(image)
        cv.imshow('blue image', blue) 
        out_b = convulation(kernel, blue, c_x, c_y)
        cv.waitKey(0)
        cv.imshow('green image', green) 
        out_g = convulation(kernel, green, c_x, c_y)
        cv.waitKey(0)
        cv.imshow('red image', red)
        out_r = convulation(kernel, red, c_x, c_y)
        cv.waitKey(0)
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        (blue_hsv , green_hsv , red_hsv) = cv.split(hsv_image)

        cv.imshow('color image', image)
        cv.imshow('hsv image', hsv_image)
        cv.waitKey(0)
        cv.imshow('blue hsv', blue_hsv) 
        out_b_h = convulation(kernel, blue_hsv, c_x, c_y)
        cv.waitKey(0)
        cv.imshow('green hsv', green_hsv) 
        out_g_h = convulation(kernel, green_hsv, c_x, c_y)
        cv.waitKey(0)
        cv.imshow('red hsv', red_hsv) 
        out_r_h = convulation(kernel, red_hsv, c_x, c_y)
        
        diff_b = out_b - out_b_h
        cv.imshow('blue diff', diff_b)
        cv.waitKey(0)
        diff_g = out_g - out_g_h
        cv.imshow('green diff', diff_g)
        cv.waitKey(0)
        diff_r = out_r - out_r_h
        cv.imshow('red diff', diff_r)
        cv.waitKey(0)
        
        color_merge = cv.merge((out_b, out_g, out_r))
        cv.imshow('Merged Color', color_merge)
        hsv_merge = cv.merge((out_b_h, out_g_h, out_r_h))
        cv.imshow('Merged HSV', hsv_merge)
        diff_merge1 = color_merge - hsv_merge
        diff_merge2 = cv.merge((diff_r, diff_g, diff_b))
        cv.normalize(diff_merge2, diff_merge2, 0, 255, cv.NORM_MINMAX)
        cv.imshow('Merged Difference1', diff_merge1)
        cv.imshow('Merged Difference2', diff_merge2)
        
        cv.waitKey(0)
        cv.destroyAllWindows()
        img_type=1
        return image
    elif inp !="":
        print("\n") 
        

def convulation(kernel, image, c_x, c_y):
    w = kernel.shape[0]//2
    h = kernel.shape[1]//2
    
    pad_bottom = kernel.shape[0] - c_x -1
    pad_right = kernel.shape[1] - c_y -1
    
    img_bordered = cv.copyMakeBorder(src=image, top=c_x, bottom=pad_bottom, left=c_y, right=pad_right,  borderType=cv.BORDER_CONSTANT)
    cv.imshow("Input Image Bordered", img_bordered)

    out = np.zeros((image.shape[0]+2*w, image.shape[1]+2*h))

    for i in range(c_x, img_bordered.shape[0] - pad_bottom - w):
        for j in range(c_x, img_bordered.shape[1] - pad_right - h):
            sum = 0
            for x in range(-w, w + 1):
                for y in range(-h, h + 1):
                    sum += kernel[x + w, y + h] * img_bordered[i - x, j - y]
            out[i, j] = sum    

    cv.normalize(out, out, 0, 255, cv.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    cv.imshow("Output Image", out) 
         
    cv.waitKey(0)
    cv.destroyAllWindows() 
    return out
        


inp = 1
while inp:
    print ("""
    1.Gaussian Kernel
    2.Mean Kernel
    3.Laplacian Kernel
    4.LoG Kernel
    5.Horizontal Sobel Kernel
    6.Vertical Sobel Kernel
    7.Exit/Quit
    """)
    inp=input("Select Kernel: ") 
    if inp=="1": 
        print("\n Gaussian Kernel") 
        kernel = gaussian_kernel()
        print(kernel)
        image = img_choice(kernel)
        # out = convulation(kernel,image, c_x, c_y)
    elif inp=="2":
        print("\n Mean Kernel") 
        kernel = mean_kernel()
        print(kernel)
        image = img_choice(kernel)
    elif inp=="3":
        print("\n Laplacian Kernel") 
        kernel = laplace_kernel_3()
        print(kernel)
        image = img_choice(kernel)
    elif inp=="4":
        print("\n LoG Kernel")
        # kernel = h_sobel_kernel()
        # print(kernel)
    elif inp=="5":
        print("\n Sobel Kernel")
        kernel = h_sobel_kernel()
        print(kernel)
        image = img_choice(kernel)
    elif inp=="6":
        print("\n Sobel Kernel")
        kernel = v_sobel_kernel()
        print(kernel)
        image = img_choice(kernel)
    elif inp=="7":
      print("\n Goodbye") 
      break
    elif inp !="":
      print("\n Try again") 
    sigma_x = float(input("Enter sigma_x: "))
    sigma_y = float(input("Enter sigma_y: "))
    k_row = int(5*sigma_x)
    k_col = int(5*sigma_y)

    c_x = int(input("Enter center x: "))
    c_y = int(input("Enter center y: "))

    if k_row%2==0:
        k_row=k_row+1       #for center symmetric
    if k_col%2==0:
        k_col=k_col+1       #for center symmetric
      
