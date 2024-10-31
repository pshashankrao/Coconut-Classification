import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import math
import matplotlib.pyplot as plt
import os
import pandas as pd

def draw_curve(p1, p2):
    if((np.cosh(p2[0]) - np.cosh(p1[0]))>0):
        a = (p2[1] - p1[1]) / (np.cosh(p2[0]) - np.cosh(p1[0]))
        b = p1[1] - a * np.cosh(p1[0])
        x = np.linspace(p1[0], p2[0], 100)
        y = a * np.cosh(x) + b
        return x, y
    else:
        return [0,0]

dataset='F:/Coco/dataset/'
#dataset='D:/Coconut/dataset/final coconut 1/TEST/'
#dataset='D:/Coconut/dataset/edit/edit'

print(os.listdir(dataset))
columns=['IMAGE_NAME','AREA','CLASS']
df=pd.DataFrame()
fname =[]
area=[]
clname=[]
for classdir in os.listdir(dataset):
  classname= classdir
  class_dir_path= os.path.join(dataset, classname)
  if(classdir != 'TEST'):
      for filename in os.listdir(os.path.join(dataset, classname)):
            filename=filename
            img = cv2.imread(os.path.join(class_dir_path, filename))
            #print(img.shape)
            #Invert binary image
            gray= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            ret, bw=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            kernel = np.ones((100, 100), np.uint8)
            imagem = cv2.bitwise_not(bw)
            imgBin = cv2.bitwise_not(imagem)
            closing = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, kernel)
            blur_img = cv2.GaussianBlur(closing, (1,1), 0)
            edges= cv2.Canny(blur_img, 30,200)
            #2. contour
            cont, hier = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #print (cont)
            #3. get large contours from image
            large_contours = []

            for k in cont:
                 if(cv2.contourArea(k) >0):
                    large_contours.append(k)

            #Draw exact contours on whiteboard
            whiteBoard = np.full(shape=img.shape, fill_value=20, dtype=np.uint8)
            cv2.drawContours(whiteBoard, large_contours, -1,255, 1)

            #create Gauss filter
            sig = 50

            #draw smooth contour
            for cnt in large_contours:
                originalX = np.array(cnt[:, 0,1])
                originalY = np.array(cnt[:, 0,0])

                #Smooth points
                xscipy = gaussian_filter(originalX, sig)
                yscipy = gaussian_filter(originalY, sig)

                #Draw smoothed points
                whiteBoard[xscipy, yscipy,  1] = 255
                # print('xscipy ',xscipy)
                # print('yscipy ',yscipy)

                #print(whiteBoard)


            left = tuple(large_contours[0][large_contours[0][:, :, 0].argmin()][0])
            right = tuple(large_contours[0][large_contours[0][:, :, 0].argmax()][0])
            extTop = tuple(large_contours[0][large_contours[0][:, :, 1].argmin()][0])
            extBot = tuple(large_contours[0][large_contours[0][:, :, 1].argmax()][0])
            cv2.line(img, left, right, (0, 255, 0), thickness=2)
            # print ("left",left)
            # print ("right",right)
            # print ("extTop",extTop)
            # print ("extBot",extBot)
            # plt.imshow(img)
            # plt.show()
            # cv2.imshow('ag', whiteBoard)
            # cv2.imshow('ag1', img)
            x,y=draw_curve((xscipy[0],yscipy[0]),(xscipy[xscipy.size-1],yscipy[yscipy.size-1]))
            min_val_index_y = yscipy.argmin()
            max_val_index_y = yscipy.argmax()
            #print ("min_val_y",yscipy[min_val_y])
            #print ("max_val_y",yscipy[max_val_y])
            #cv2.line(img, (xscipy[min_val_index_y],yscipy[min_val_index_y]), (xscipy[max_val_index_y],yscipy[max_val_index_y]), (0, 255, 0), thickness=2)

            plt.subplots()
            plt.plot([xscipy[min_val_index_y],xscipy[max_val_index_y]],[yscipy[min_val_index_y],yscipy[max_val_index_y]])
            plt.plot(originalX,originalY)
            plt.plot(xscipy, yscipy,color="#ff7f0e")
            plt.plot(x,y)
            plt.title(classname+'......'+filename, loc = 'left')

            #plt.show(x,y)
            # print('x', x)
            # print('y',y)
            # print ("xscipy[min_val_index_y]",xscipy[min_val_index_y])
            # print ("yscipy[min_val_index_y]",yscipy[min_val_index_y])
            # print ("xscipy[max_val_index_y]",xscipy[max_val_index_y])
            # print ("yscipy[max_val_index_y]",yscipy[max_val_index_y])
            x_mid_end=(xscipy[0]+xscipy[xscipy.size-1])/2
            y_mid_end=(yscipy[0]+yscipy[yscipy.size-1])/2
            x_mid_minor=(xscipy[min_val_index_y]+xscipy[max_val_index_y])/2
            y_mid_minor=(yscipy[min_val_index_y]+yscipy[max_val_index_y])/2
            plt.plot([x_mid_end,x_mid_minor],[y_mid_end,y_mid_minor])

            major_axis_dist=math.dist([x_mid_end,x_mid_minor], [y_mid_end,y_mid_minor])*2
            minor_axis_dist=math.dist([xscipy[min_val_index_y],xscipy[max_val_index_y]],[yscipy[min_val_index_y],yscipy[max_val_index_y]])
           # print('major_axis_dist',major_axis_dist)
            #print('minor_axis_dist',minor_axis_dist)
            area_ellipse=round(math.pi*major_axis_dist*minor_axis_dist)

            print('IMAGENAME',filename)
            print('AREA********************************************', area_ellipse)
            print('CLASS',classname)
            fname.append(filename)
            area.append(area_ellipse)
            clname.append(classname)
            #dic_dat= pd.concat(df,pd.DataFrame['IMAGENAME':filename,'AREA':area_ellipse,'CLASS':classname])
            #df=pd.DataFrame([filename,area_ellipse,classname],index=['IMAGENAME','AREA','CLASS']).T
            #df.merge(pd.DataFrame['IMAGENAME':filename,'AREA':area_ellipse,'CLASS':classname],left_index=True, right_index=True)
df=pd.DataFrame([fname,area,clname],index=['IMAGENAME','AREA','CLASS']).T
df.to_excel('area_class.xlsx', sheet_name='sheet1', index=True)
cv2.waitKey(0)
cv2.destroyAllWindows()
