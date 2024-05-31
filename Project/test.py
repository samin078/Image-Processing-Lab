from filter import *
import numpy as np
import cv2

sigma_x = 1
sigma_y = 1

kernel = gaussian_kernel(sigma_x, sigma_y)

img = img = cv2.imread("five.jpeg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img , (640 , 800))

blurred_img = convolution(kernel, img)

def empty(a):
    pass

# Create a window for trackbars
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Threshold1", "Settings", 175, 255, empty)
cv2.createTrackbar("Threshold2", "Settings", 210, 255, empty)

while True:
    # Get the current positions of the trackbars
    thresh1 = cv2.getTrackbarPos("Threshold1", "Settings")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Settings")
    
    # Print the current values of the trackbars
    print(thresh1, thresh2)
    
    # Wait for a short period of time and check for the 'q' key to exit
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break


ret , thresh = cv2.threshold(blurred_img , thresh1 , thresh2 , cv2.THRESH_BINARY)

contours , _ = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)
area = {}
for i in range(len(contours)):
    cnt = contours[i]
    ar = cv2.contourArea(cnt)
    area[i] = ar
srt = sorted(area.items() , key = lambda x : x[1] , reverse = True)
results = np.array(srt).astype("int")
num = np.argwhere(results[: , 1] > 500).shape[0]

for i in range(1 , num):
    blurred_img = cv2.drawContours(blurred_img , contours , results[i , 0] ,
                                  (0 , 255 , 0) , 3)
print("Number of coins is " , num - 1)
cv2.imshow("final" , blurred_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
# # cv2.imshow("Inp", img)
# # cv2.imshow("Out", output_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()