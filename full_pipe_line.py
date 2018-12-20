
import numpy as np
import pandas as pd
import cv2
import math
import os
import argparse
import pytesseract
import operator
from PIL import Image
from darknet import *
from tqdm import tqdm



net = load_net(b"/home/sasuke/Downloads/All_detection/yolov3-table.cfg", 
               b"/home/sasuke/Downloads/All_detection/yolov3-table_18000.weights", 0)
meta = load_meta(b"/home/sasuke/Downloads/All_detection/data/table.data")


#image_path = b"/home/sasuke/Downloads/darknet-master/test_data/time.png"


def cropping(image_path_str, r):
    
    for i in range(len(r)):
        a , b , c , d   = r[i][2]
        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)
        img = cv2.imread(image_path_str)
        img1 = img[b-d//2:b+d//2, a-c//2:a+c//2]
        img1 = cv2.resize(img1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        q = cv2.imwrite('cropped_images/' + image_path_str.split('/')[-1].split('.')[0] + '_' + str(i+1) + '.jpg', img1)
    
    #return q


def sort_contours(cnts, method="left-to-right"):
   # initialize the reverse flag and sort index
   reverse = False
   i = 0

   # handle if we need to sort in reverse
   if method == "right-to-left" or method == "bottom-to-top":
       reverse = True

   # handle if we are sorting against the y-coordinate rather than
   # the x-coordinate of the bounding box
   if method == "top-to-bottom" or method == "bottom-to-top":
       i = 1

   # construct the list of bounding boxes and sort them from top to
   # bottom
   boundingBoxes = [cv2.boundingRect(c) for c in cnts]
   (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
       key=lambda b:b[1][i], reverse=reverse))

   # return the list of sorted contours and bounding boxes
   return (cnts, boundingBoxes)



def row_detect(give_images):
    
    
    img = cv2.imread(give_images, 0)
 
    # Thresholding the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # Invert the image
    img_bin = 255-img_bin 

    #cv2.imwrite("Image_bin_asd.jpg",img_bin)
    
    #print('.....')
    kernel_length = np.array(img).shape[1]//50
    kernel_length_ver = np.array(img).shape[1]//30

    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_ver))

    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
   # print('=====')
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    #cv2.imwrite("verticle_lines_fuch.jpg",verticle_lines_img)
    
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    #cv2.imwrite("horizontal_lines_fuch.jpg",horizontal_lines_img)
    
    
    alpha = 0.5
    beta = 1.0 - alpha
    
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imwrite("img_final_bin_image_44300.v1.jpg",img_final_bin)
    
    im2, contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    #print('++++')
    idx = 0
    l = []

    for c in contours:
       # pint('^^^^')
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        # print(x);print(y);print(w);print(h);print('....')
        if (w > 80 and h > 20) and w > 1*h:
            idx += 1
            new_img = img[y:y+h, x:x+w]
            #print('******')
            cv2.imwrite('kaata/' + give_images.split('/')[-1].split('.')[0] + '_' + str(idx) + '.jpg', new_img)
            
            im = Image.open('kaata/' + give_images.split('/')[-1].split('.')[0] + '_' + str(idx) + '.jpg')
            nx, ny = im.size
            im = im.resize((int(nx*2.5), int(ny*2.5)), Image.BICUBIC)
            im.save("resized/resize_" + give_images.split('/')[-1].split('.')[0] + '_' + str(idx) + '.jpg'
                            , dpi=(300,300))
            
            #ghaat = "resized/resize_" + give_images.split('/')[-1].split('.')[0] + '_' + str(idx) + '.jpg'
            #print(ghaat)
            
            #chal = give_images.split('/')[-1].split('.')[0] + '_' + str(idx)
            text = pytesseract.image_to_string(Image.open("resized/resize_" + 
                                                          give_images.split('/')[-1].split('.')[0] + '_' + 
                                                          str(idx) + '.jpg'))
            
            if text == (''):
                pass 
            
            else:
                l.append(text)
                
    
    df_l = pd.DataFrame(l)
   # print(df_l)
    
            
    return df_l

        

def sakta(image_path):
    bob = []
    r = detect(net, meta, image_path)
    r = sorted(r,key = operator.itemgetter(2))
    image_path_str = image_path.decode("utf-8")
    h = cropping(image_path_str, r)
    
    for i in tqdm(range(len(r))):
        give_images = 'cropped_images/' + image_path_str.split('/')[-1].split('.')[0] + '_' + str(i+1) + '.jpg'
        #print(give_images)
        y = row_detect(give_images)
        bob.append(y)
        #y = pd.DataFrame(bob)
        y = pd.concat(bob, axis=1)
        #y = y.rename(columns=y.iloc[0]).drop(y.index[0])
        
    #print(y.iloc[0])
    #y = y.rename(columns=y.iloc[0]).drop(y.index[0])
    
    new_header = y.iloc[0] # the first row for the header
    y = y[1:] #take the data less the header row
    y.columns = new_header
    #print(y)
    #y = y.reset_index(drop = True)
    y = y.to_csv('csv_results/' + image_path_str.split('/')[-1].split('.')[0] + '.csv', index = None)
    
    return y


#df = pd.read_csv('csv_results/time.csv')


# In[ ]:
# def main(args):

#   sakta(args.image_path)


if __name__ == "__main__":

  # parser = argparse.ArgumentParser()

  # parser.add_argument('--image_path', type = bytes, help='path for image')

  # args = parser.parse_args()
  # print(args)
  # main(args)

  image_path = b"/home/sasuke/Downloads/All_detection/test_data/time.png"
  j = sakta(image_path)
  


#kar do changes
#nhi krenge


    

