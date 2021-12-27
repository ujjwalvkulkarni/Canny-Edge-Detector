# CSGY 6643 - Computer Vision Project 1 - Canny Edge Detector
# Completed by group of 2:
#   1. Harsh Sonthalia - hs4226
#   2. Ujjwal Vikram Kulkarni - uk2011
    
import math
import os
import cv2
import numpy as np
from cv2 import imwrite

#The location of the image whose edge is to be detected
image_path = '/Users/ujjwalkulkarni/Desktop/House.bmp'

#The location of the folder where the output images are to be stored
output_path = '/Users/ujjwalkulkarni/Desktop/Canny/HouseTest'

# Designing a Canny Edge detector consits of four functions - Gaussian Smoothening, Gradient Magnitude & Angle, Non-Maxima Supression and Thresholding.
# These four functions along with the main function 'Canny' have been displayed below

#removing noise by filtering the image with a gaussian filter
def GaussianSmoothening(image):
    gaussian_filter =  np.array([[1, 1, 2, 2, 2, 1, 1],
                                 [1, 2, 2, 4, 2, 2, 1],
                                 [2, 2, 4, 8, 4, 2, 2],
                                 [2, 4, 8, 16, 8, 4, 2],
                                 [2, 2, 4, 8, 4, 2, 2],
                                 [1, 2, 2, 4, 2, 2, 1],
                                 [1, 1, 2, 2, 2, 1, 1]])
    
    # The shape of the image is [225,225,4] which is converted to 2D by just considering the Height and Width
    height = image.shape[0]
    width = image.shape[1]
    
    #creating a variable to store output image after applying gaussian filter
    convolved_gaussian_array = np.empty((height,width))
    
    # Since the Gaussian filter is a 7x7 filter with the center as the reference point, the first if statement assigns 0 to all values that are undefined.
    
    for row in range(height):
        for col in range(width):
            
            if 0 <= row < 3 or height - 3 <= row < height or 0 <= col < 3 or width - 3 <= col < width:
                convolved_gaussian_array[row][col] = 0
                
            else:
                sum = 0
                
                # This loop convoles the Gaussian filter with the given Image 
                
                for i in range(7):
                    for j in range(7):
                        sum = sum + (image[row - 3 + i][col - 3 + j]) * gaussian_filter[i][j]
                        
                # Normalizing the Gaussian Smoothened Image by 140
                
                convolved_gaussian_array[row][col] = sum/140                        
    
    #saving gaussian output image in the output folder
    imwrite(os.path.join(output_path, "House_gaussiansmoothing.bmp"), convolved_gaussian_array)
    return convolved_gaussian_array
    
    

#computing gradient magnitude and gradient angle using Prewitt's operator    
def Prewitt(smoothedImage):
    prewittX = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])

    prewittY = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])
    
    height = smoothedImage.shape[0]
    width = smoothedImage.shape[1]
    
    # We define two 2D arrays for the horizontal and vertical gradient
    
    horizontal_grad = np.empty((height, width))
    vertical_grad = np.empty((height, width))
    
    # In this loop, we know that there are a total of 6 rows and columns that are already undefined. Since the Pwreitt filter will also go out of bounds, we increase the dimention to 4
    # This is done for both, horizontal and vertical gradient
    
    for row in range(height):
        for col in range(width):
            if 0 <= row < 4 or height - 4 <= row < height or 0 <= col < 4 or width - 4 <= col < width:
                vertical_grad[row][col] = 0
                
                
            # Calculating the Vertical Gradient of Prewitt filter with the Gaussian Smoothened Image
            
            else:
                sum = 0
                for i in range(3):
                    for j in range(3):
                        
                        sum = sum + (smoothedImage[row - 1 + i][col - 1 + j]) * prewittX[i][j]
                
                vertical_grad[row][col] = sum
                
    for row in range(height):
        for col in range(width):
            if 0 <= row < 4 or height - 4 <= row <= height - 1 or 0 <= col < 4 or width - 4 <= col <= width - 1:
                horizontal_grad[row][col] = 0
                
            # Calculating the Horizontal Gradient of Prewitt filter with the Gaussian Smoothened Image
            
            else:
                sum = 0
                for i in range(3):
                    for j in range(3):
                        sum = sum + (smoothedImage[row - 1 + i][col - 1 + j]) * prewittY[i][j]
                horizontal_grad[row][col] = sum
                
    #Taking absolute value to remove negative values and normalising the gradient values by a factor of 3           
    vertical_grad_abs = np.true_divide(np.abs(vertical_grad),3)
    horizontal_grad_abs = np.true_divide(np.abs(horizontal_grad),3)
                
    #saving gradient images in the output folder
    imwrite(os.path.join(output_path, "House_horizontalGradient.bmp"), horizontal_grad_abs)
    imwrite(os.path.join(output_path, "House_verticalGradient.bmp"), vertical_grad_abs)
    
    # We create a third 2D array to calculate the Gradient Magnitude
    
    grad_magnitude = np.empty((height, width))
    
    # The absolute values of the horizonatal and vertical gradient are used to calculate the gradient magnitude
    # Grad(x,y) = |x| + |y|, where x and y are the Horizontal and Vertical Gradients respectively
    
    for k in range(height):
        for l in range(width):
            
            grad_magnitude[k][l] =  abs( horizontal_grad_abs[k][l] ) + abs( vertical_grad_abs[k][l] )
    
    #Normalizing the Gradient Magnitude by a factor of 2
    grad_magnitude_normal = np.true_divide(grad_magnitude ,2)
    
    #Saving the Gradient Magnitude image in the output folder
    imwrite(os.path.join(output_path, "House_GradientMagnitude.bmp"), grad_magnitude_normal)
    
    # We define a fourth 2D array to calculate the gradient angle
    # Angle = inverseTan(y/x), where x and y are the Horizontal and Vertical Gradients respectively
    
    grad_angle = np.empty((height, width))
    
    for k in range(height):
        for l in range(width):
            
            # Condition to avoid Divison by 0 error
            if vertical_grad[k][l] == 0:
                grad_angle[k][l] = 0
                
            else:    
                grad_angle[k][l] = math.degrees( math.atan( horizontal_grad[k][l] / vertical_grad[k][l] ))
    
    #print(grad_angle)
    
    return grad_magnitude_normal, grad_angle
    


#Computing the Non Maxima Suppressed image
def NMS(grad_magnitude, grad_angle):
    
    height = grad_magnitude.shape[0]
    width = grad_magnitude.shape[1]
    
    nms = np.empty((height, width))
    sector = np.empty((height, width))

    # The loop ranges to height-4 and width-4 in order to avoid values that are out of bounds to calculate sector
    
    for k in range(3, height-4):
        for l in range(3, width-4):
            #converting all negative angles to positive angles
            if grad_angle[k][l] < 0:
                grad_angle[k][l] = 360 + grad_angle[k][l]
            #Check if the grad angle lies is the 0th sector    
            if 0 <= grad_angle[k][l] < 22.5 or  337.5 <= grad_angle[k][l] <= 360 or  157.5 <= grad_angle[k][l] < 202.5 :
                sector[k][l] = 0
            #Check if the grad angle lies is the 1st sector         
            elif 22.5 <= grad_angle[k][l] < 67.5 or  202.5 <= grad_angle[k][l] < 247.5:
                sector[k][l] = 1
                    
            #Check if the grad angle lies is the 2nd sector        
            elif 67.5 <= grad_angle[k][l] < 112.5 or  247.5 <= grad_angle[k][l] < 292.5:
                sector[k][l] = 2

            #Check if the grad angle lies is the 3rd sector        
            else:
                sector[k][l] = 3

    # The loop to calculate non maxima supressed magnitide
    for k in range(3, height-4):
        for l in range(3, width-4):

            #the grad magnitude for the 0th sector    
            if sector[k][l] == 0 :
                if grad_magnitude[k][l] > grad_magnitude[k][l-1] and grad_magnitude[k][l] > grad_magnitude[k][l+1]:
                    nms[k][l] = grad_magnitude[k][l]
                else:
                    nms[k][l] = 0 
            #the grad magnitude for the 1st sector         
            elif sector[k][l] == 1 :
                if grad_magnitude[k][l] > grad_magnitude[k-1][l+1] and grad_magnitude[k][l] > grad_magnitude[k+1][l-1]:
                    nms[k][l] = grad_magnitude[k][l]
                else:
                    nms[k][l] = 0
                    
            #the grad magnitude for the 2nd sector        
            elif sector[k][l] == 2 :
                if grad_magnitude[k][l] > grad_magnitude[k-1][l] and grad_magnitude[k][l] > grad_magnitude[k+1][l]:
                    nms[k][l] = grad_magnitude[k][l]
                else:
                    nms[k][l] = 0

            #the grad magnitude for the 3rd sector        
            else:
                if grad_magnitude[k][l] > grad_magnitude[k-1][l-1] and grad_magnitude[k][l] > grad_magnitude[k+1][l+1]:
                    nms[k][l] = grad_magnitude[k][l]
                else:
                    nms[k][l] = 0


    imwrite(os.path.join(output_path, "House_NMS.bmp"), nms)

    #print(nms)            
    
    return nms
                
                
    
#Threhsolding the nms image to find edges    
def Threshold(nms):
    
    height = nms.shape[0]
    width = nms.shape[1]
    
    # Three separate 2D arrays in order to store images with three separate threshold percentiles - 25th, 50th and 75th
    
    Threshold_25 = np.empty((height, width))
    Threshold_50 = np.empty((height, width))
    Threshold_75 = np.empty((height, width))
    
    
    # np.percentile used to find the percentile after excluding 0. 
    
    t_25 = np.percentile(nms[nms!=0], 25)
    t_50 = np.percentile(nms[nms!=0], 50)
    t_75 = np.percentile(nms[nms!=0], 75)

    for k in range(height):
        for l in range(width):
            
            #Assigning 25th percentile values
            if nms[k][l] >= t_25:
                Threshold_25[k][l] = 0    
            else:    
                Threshold_25[k][l] = 255
            
            #Assigning 50th percentile values
            if nms[k][l] >= t_50:
                Threshold_50[k][l] = 0    
            else:    
                Threshold_50[k][l] = 255  

            #Assigning 75th percentile values
            if nms[k][l] >= t_75:
                Threshold_75[k][l] = 0    
            else:    
                Threshold_75[k][l] = 255       

    imwrite(os.path.join(output_path, "House_Threshold at 25.bmp"), Threshold_25)
    imwrite(os.path.join(output_path, "House_Threshold at 50.bmp"), Threshold_50)
    imwrite(os.path.join(output_path, "House_Threshold at 75.bmp"), Threshold_75)
    return Threshold_25, Threshold_50, Threshold_75
    
    
def Canny():
    image = cv2.imread(image_path, 0)
    
    gaussian_output = GaussianSmoothening(image)
    
    prewitt_output, gradientAngle = Prewitt(gaussian_output)
    
    NMS_output = NMS(prewitt_output, gradientAngle)
    
    Threshold_output25, Threshold_output50, Threshold_output75 = Threshold(NMS_output)
    
    return
    

Canny()
    
    

    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    

        

