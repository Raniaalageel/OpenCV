# By Rania Alageel 
import cv2
import numpy as np
from datetime import datetime, timedelta
start = datetime.now()
cpt = 0
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

# haarcascade

# face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# def detect_faces(gray , frame):
    
#     """
#     This function for detect faces 
#     1- detect faces in image
#     2- draw blue rectangle around the face 
#     """

#     faces = face_detect.detectMultiScale(gray,1.3 ,5)

#     # (x , y , w , h)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x,y), (x+w,y+h) , (255,0,0) , 2)

#     return frame

# def RGB2HEX(color):
#     return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


# def Color_filtering(image):
#  print("The type of this input is {}".format(type(image)))
#  print("Shape: {}".format(image.shape))
#  cv2.imshow('Actual image',image)
#  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#  cv2.imshow('COLOR_BGR2RGB',image)
#  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #  cv2.imshow(gray_image, cmap='gray')
#  cv2.imshow('gray_image',gray_image)
 


# def get_image(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image


# def get_colors(image,number_of_colors,show_chart):
 
#  print("inter get color")
#  modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
#  modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
#  clf = KMeans(n_clusters = number_of_colors)
#  labels = clf.fit_predict(modified_image)
#  counts = Counter(labels)
#  center_colors = clf.cluster_centers_
#  # We get ordered colors by iterating through the keys
#  ordered_colors = [center_colors[i] for i in counts.keys()]
#  hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
#  rgb_colors = [ordered_colors[i] for i in counts.keys()]

#  print("out of get color")

#  if (show_chart):
#     print("inter pie chart")
#     plt.figure(figsize = (8, 6))
#     plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
#     plt.show()
#     print("out of pie chart")
#  return rgb_colors


img = cv2.imread('group5.jpg')
image = cv2.imread('group5.jpg')


class ImageProcessor():
  

# calculate_histogram :
  def calculate_histogram(image):

    #  Gray-scale :
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [255], [0, 255])
    plt.plot(histogram, color='k')
    plt.suptitle('Gray scale histogram')
    plt.show()

    # Color image : 
    for i, col in enumerate(['b', 'g', 'r']):
      hist = cv2.calcHist([image], [i], None, [255], [0, 255])
      plt.plot(hist, color = col)
      plt.suptitle('Color histogram')
      plt.xlim([0, 255])
    plt.show() 

# equalize_histogram : 
  def equalize_histogram(image):
   
   # Gray-scale :
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_grayscale_image = cv2.equalizeHist(grayscale_image)
    # plt.imshow(eq_grayscale_image, cmap='gray')
    # plt.suptitle('equalize Gray scale image')
    # plt.show()
    histogram = cv2.calcHist([eq_grayscale_image], [0], None, [255], [0, 255])
    plt.plot(histogram, color='k')
    plt.suptitle('Gray scale equalized image histogram')
    plt.show()
    
    # !!!!!!!!!!!! Color image : 
    channels = cv2.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
      eq_channels.append(cv2.equalizeHist(ch))
    eq_image = cv2.merge(eq_channels)
    eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB)
    # plt.suptitle('Color equalized image')
    # plt.imshow(eq_image)
    # plt.show()

    # show Histogram
    channels = ('b', 'g', 'r')
    # we now separate the colors and plot each in the Histogram
    for i, color in enumerate(channels):
      histogram = cv2.calcHist([eq_image], [i], None, [255], [0, 255])
      plt.plot(histogram, color=color)
      plt.xlim([0, 255])
    plt.suptitle('Color equalized image histogram')
    plt.show()
    # !!!!!!!!!!!! Color image : 


  def Display_Images(image):
   

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_grayscale_image = cv2.equalizeHist(grayscale_image)
    Original_image = image
    channels = cv2.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
      eq_channels.append(cv2.equalizeHist(ch))
    eq_image = cv2.merge(eq_channels)
    eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB)



    # create figure
    fig = plt.figure(figsize=(10, 7))
  
    # setting values to rows and column variables
    rows = 2
    columns = 2
  
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
  
    # showing image
    plt.imshow(grayscale_image, cmap='gray')
    plt.title("Original gray scale image")
  
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
  
    # showing image
    plt.imshow(eq_grayscale_image, cmap='gray')
    plt.title("Equalized gray scale image")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
  
    # showing image
    plt.imshow(image[:, :, ::-1])
    # imgggg = cv2.imread('group5.jpg')
    # plt.imshow(imgggg[:, :, ::-1])
    plt.title("Original color image")
  
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)
  
    # showing image
    plt.imshow(eq_image)
    plt.title("Equalized color image")

   

class Derived(ImageProcessor):
   ImageProcessor.calculate_histogram(image)
   ImageProcessor.equalize_histogram(image)
   ImageProcessor.Display_Images(image)





face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_faces(gray , frame):
    
    """
    This function for detect faces 
    1- detect faces in image
    2- draw blue rectangle around the face 
    """

    faces = face_detect.detectMultiScale(gray,1.3 ,5)

    # (x , y , w , h)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h) , (255,0,0) , 2)

    return frame

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def Color_filtering(image):
 print("The type of this input is {}".format(type(image)))
 print("Shape: {}".format(image.shape))
#  cv2.imshow('Actual image',image)
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#  cv2.imshow('COLOR_BGR2RGB',image)
 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#  cv2.imshow(gray_image, cmap='gray')
#  cv2.imshow('gray_image',gray_image)
 
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_colors(image,number_of_colors,show_chart):
 
 print("inter get color")
 modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
 modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
 clf = KMeans(n_clusters = number_of_colors)
 labels = clf.fit_predict(modified_image)
 counts = Counter(labels)
 center_colors = clf.cluster_centers_
 # We get ordered colors by iterating through the keys
 ordered_colors = [center_colors[i] for i in counts.keys()]
 hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
 rgb_colors = [ordered_colors[i] for i in counts.keys()]

 print("out of get color")

 if (show_chart):
    print("inter pie chart")
    plt.figure(figsize = (8, 6))
    plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    plt.show()
    print("out of pie chart")
 return rgb_colors



# get_colors(get_image('group.jpg'), 8, True)
Filter_Colors = Color_filtering(image)
get_colors(get_image('group.jpg'), 8, True)
 

# The Image Processor class should have a method calculate histogram to calculate the histogram of the input image.
# def calculate_histogram(image2):
 
#  for i, col in enumerate(['b', 'g', 'r']):
#     hist = cv2.calcHist([image2], [i], None, [256], [0, 256])
#     plt.plot(hist, color = col)
#     plt.xlim([0, 256])
    
#  plt.show()


# image2 = cv2.imread('image3.jpeg')
# calculate_histogram(image2)

# The ImageProcessor class should have a method equalize histogram to perform histogram equalization on the input image.
# def equalize_histogram(image):
#    channels = cv2.split(image)
#    eq_channels = []
#    for ch, color in zip(channels, ['B', 'G', 'R']):
#              eq_channels.append(cv2.equalizeHist(ch))

#    eq_image = cv2.merge(eq_channels)
#    eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB)

#    cv2.imshow("Original", image)
#    cv2.imshow("Equalized Image", eq_image)
# #    plt.imshow(eq_image)

#   # show Histogram
#    channels = ('b', 'g', 'r')
#   # we now separate the colors and plot each in the Histogram
#    for i, color in enumerate(channels):
#      histogram = cv2.calcHist([eq_image], [i], None, [256], [0, 256])
#      plt.plot(histogram, color=color)
#      plt.xlim([0, 256])

#    plt.show()


# image3 = cv2.imread('image3.jpeg')
# equalize_histogram(image3)

# Create a Python program that uses OpenCV's Haar cascades to detect faces in a webcam stream. Draw rectangles around the detected faces.
# Create a Python program that loads an image and converts it to different color spaces, such as HSV and LAB.
while True:
    
    # _, frame = cap.read()
    

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect = detect_faces(gray, frame)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect = detect_faces(gray, img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      
    # Threshold of blue in HSV space
    lower_blue = np.array([60, 35, 140])
    upper_blue = np.array([180, 255, 255])
  
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
      
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(img, img, mask = mask)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # while datetime.now() - start < timedelta(seconds=5*60*60):        
     # Shows the image
    cv2.imshow('imageLAB', lab) 
    cv2.imshow('imageHSV', hsv) 
    cv2.imshow('Actual image' , detect)
    # cv2.imshow('Masked Image ', mask)
    # cv2.imshow('Blue Color segmented regions', result)
    # cpt += 1
    # print(cpt)  
    # while(True):
    #     k = cv2.waitKey(3000)
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows() 
 
    # # Shows the image
    # cv2.imshow('imageLAB', lab) 
    # cv2.imshow('imageHSV', hsv) 
    # cv2.imshow('Actual image' , detect)
    # cv2.imshow('Masked Image ', mask)
    # cv2.imshow('Blue Color segmented regions', result)
    
    # image = cv2.imread('image4.jpg')
    # Filter_Colors = Color_filtering(image)


    # esc
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break



cv2.waitKey(0)         
cv2.destroyAllWindows()




