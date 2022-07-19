import cv2
import imutils
import numpy as np
import os
from typing import List
import argparse

# usage: python image_extractor <path/to/folder>

# imagefile extensions
FILE_EXTENSIONS: List[str] = [".jpg", ".tiff", ".png"]


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(dest='path', help="Path of the images", default=".")
parser.add_argument('-s',help="Show detected photos", action="store_true", default=False)
args = parser.parse_args()
path = args.path


# check for for filetype: image
def check_file_extension(filename: str) -> bool:
    for e in FILE_EXTENSIONS:
        if(filename.upper().endswith(e.upper())):
            return True
    return False

def show_image(image):
        image = imutils.resize(image, height = 800)
        cv2.imshow("Image", image)
        print("Press any key to continue...")
        cv2.waitKey(0)

i = 0
for file in os.listdir(path):
    target = os.path.join(path,"processed/")
    if check_file_extension(file):         # check if file is an image
        if not os.path.exists(target):
            os.makedirs(target)
        img_path = os.path.join(path,file)
        print(img_path)

        # read the image
        image = cv2.imread(img_path)

        # trim the border: prevent articfacts from scanner
        bordersize = 10
        y = image.shape[0]
        x = image.shape[1]
        if y > x:
         image = image[20:y-40, 5:x-5]

        # pad the image
        image = cv2.copyMakeBorder(image, 100,100,100,100,cv2.BORDER_CONSTANT,None,(255,255,255))


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        ret, th = cv2.threshold(gray, 210, 235, 1)

        # find contours on scanned image
        cnts, hierarchy = cv2.findContours(
            th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        i = 0
        if(args.s):
            drawing_copy = image.copy()

        for c in cnts:
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(
                box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            area = image.shape[0]*image.shape[1]

            # filter small areas
            if area/15 < cv2.contourArea(box):
                x, y, width, height = cv2.boundingRect(c)
                roi = image[y:y+height, x:x+width] # cut out region of interest
                file_write_path = os.path.join(target,str(i)+"-"+file)
                print(file_write_path)
                try:
                    cv2.imwrite(file_write_path, roi)
                    if(args.s):
                        cv2.drawContours(drawing_copy, [box], -1, (0, 255, 0), 5)
                    i += 1

                except:
                    print("Error writing file to " + file_write_path)
        # show preview
        if(args.s):
            show_image(drawing_copy)
