 # -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 08:49:32 2020

@author: hj123
"""
import cv2
import numpy as np

def displayIMG(img, windowName):
    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(windowName,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


inp_path = '\\Pics\\Quest\\R 600 L 50.png'

img = cv2.imread(inp_path)
# displayIMG(img, "Original")        

resized = cv2.resize(img, (0,0), fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
h, w, c = resized.shape
displayIMG(resized, "Resized")

center = (int(w/2), int(h/2))
cropped = resized[0:h,center[0]-center[1]:center[0]+center[1]]
displayIMG(cropped, "Cropped")  

img_gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)

# histogram stretching
img_gray_hs = 255*(img_gray.astype('float64')-img_gray.min())/(img_gray.max()-img_gray.min())
img_gray_hs = img_gray_hs.astype('uint8')
displayIMG(img_gray_hs, "img_gray_hs")

img_gray_b = cv2.GaussianBlur(img_gray_hs,(5,5),0)
edged = cv2.Canny(img_gray_b,100,250)
displayIMG(edged, "Edged")

# ret, binary = cv2.threshold(edged,127,255,cv2.THRESH_BINARY)
ret, binary = cv2.threshold(edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
displayIMG(binary, "thresholding") 

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
img_mor = cv2.dilate(binary, kernel)
displayIMG(img_mor, "MORPH_DILATE")

# img_mor = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel,iterations=1),
# displayIMG(img_mor[0], "MORPH_CLOSE")

# # # Visual studio code
# # (_, cnts, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Spyder
cnts, _ = cv2.findContours(img_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Count {} contours in this image".format(len(cnts)))
contours = cropped.copy()
cv2.drawContours(contours, cnts, -1, (0, 255, 0), 2)

pos = []
center = []
area_c = []
min_circle_in = []
radius_c = []
mask_c = []
L_c = []
for (i, c) in enumerate(cnts):
    
    (x, y, w, h) = cv2.boundingRect(c)
    pos.append([x, y, w, h])
    print("Contour #{}".format(i + 1))
    
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center.append((cX, cY))
    cv2.circle(contours, (cX, cY), 10, (1, 227, 254), -1)
    area = cv2.contourArea(c)  #calculate area of contour
    area_c.append(area)
    perimeter = cv2.arcLength(c, True)    #calculate perimeter of contour
    print("Contour #%d — area: %.2f, perimeter: %.2f" % (i + 1, area, perimeter))  #print the results
    #put the No. of contour to the image
    cv2.putText(contours, "#%d" % (i + 1), (cX-20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 50, 250), 3)
    displayIMG(contours, "Contours")
    
    # Extract countours
    contour = cropped[y:y + h, x:x + w]
    mask = np.zeros(cropped.shape[:2], dtype = "uint8")
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    min_circle_in.append((centerX, centerY))
    radius_c.append(radius)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
     
    mask = np.zeros(img_gray.shape, dtype='uint8')
    cv2.drawContours(mask, [c], -1, 255, -1)
    mask_m = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    mask_c.append(mask_m)
    displayIMG(cv2.bitwise_and(img_gray, img_gray, mask = mask), "Masked Contours")
    L = np.mean(cv2.bitwise_and(img_gray, img_gray, mask = mask))
    L_c.append(L)
 
    

if len(cnts) > 2:
    area_f = np.max(area_c)
    ind_max = np.argmax(area_c)  
    ind_min = np.argmin(area_c)
    max_c = center[ind_max]
    min_c = center[ind_min]
    d = pow((pow((max_c[0]-min_c[0]),2)+pow((max_c[1]-min_c[1]),2))/2,0.5)/2
    
    if d < 1:
        radius_f = np.max(radius_c)
        mask_a = np.array(mask_c[ind_max])
        mask_b = np.array(mask_c[ind_min])
        mask_r = mask_a - mask_b
        pixels1 = mask_r[mask_r > 0]
        n_pixels1 = pixels1.shape[0]
        displayIMG(mask_r, 'Real mask')
        L_f = np.sum(mask_r)/n_pixels1
        
        mask_1 = np.ones(img_gray.shape, dtype="uint8")  #build the mask based on contour
        mask_2 = cv2.drawContours(mask_a, cnts, -1, 0, -1)
        mask_3 = cv2.drawContours(mask_b, cnts, -1, 0, -1)
        # mask_3 = cv2.drawContours(mask_1, cnts[ind_min], -1, 0, cv2.FILLED)
        mask_4 = mask_2 - mask_3
        mask_5 = mask_4.copy()
        mask_5[mask_5 > 0] = 1
        displayIMG(mask_5*255,"Mask")
        pixels2 = mask_5[mask_r == 0]
        n_pixels2 = pixels2.shape[0]
        other = img_gray*mask_5
        other_L = np.sum(img_gray*mask_5)/n_pixels2
        p_n = np.sum(mask_3)
        flat_sort = np.sort(other.flatten())
        other_L_n0 = flat_sort[p_n::]
        other_L_n0 = other[other != 0]
        std_other_L = pow((np.sum(pow((other_L_n0-other_L),2))/n_pixels2),0.5)
    
    else:
        cnts2, _ = cv2.findContours(img_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_f = np.max(area_c)
        p = np.array(pos)
        x_n1 = np.min(p[:,0])
        y_n1 = np.min(p[:,1])
        x_n2 = np.max(p[:,0]+p[:,2])
        y_n2 = np.max(p[:,1]+p[:,3])
        radius_f = pow((pow((x_n2-x_n1),2)+pow((y_n2-y_n1),2))/2,0.5)/2
                
        mask_1 = np.ones(img_gray.shape, dtype="uint8")  #build the mask based on contour
        mask_2 = cv2.drawContours(mask_1, cnts2, -1, 0, cv2.FILLED)
        # mask_s = cv2.drawContours(mask_1, cnts, -1, 255, cv2.FILLED)
        mask_3 = 1 - mask_2
        pixels1 = mask_3[mask_3 > 0]
        n_pixels1 = pixels1.shape[0]
        L_f = np.sum(img_gray*mask_3)/n_pixels1
        displayIMG(mask_2*255,"Mask")
        pixels2 = mask_3[mask_3 == 0]
        n_pixels2 = pixels2.shape[0]
        other = img_gray*mask_2
        other_L = np.sum(img_gray*mask_2)/n_pixels2
        p_n = np.sum(mask_3)
        flat_sort = np.sort(other.flatten())
        other_L_n0 = flat_sort[p_n::]
        # other_L_n0 = other[other != 0]
        std_other_L = pow((np.sum(pow((other_L_n0-other_L),2))/n_pixels2),0.5)
        
else:
    area_f = np.max(area_c)
    radius_f = np.max(radius_c)
    cnts3, _ = cv2.findContours(img_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_1 = np.ones(img_gray.shape, dtype="uint8")  #build the mask based on contour
    mask_2 = cv2.drawContours(mask_1, cnts3, -1, 0, cv2.FILLED)
    # mask_s = cv2.drawContours(mask_1, cnts, -1, 255, cv2.FILLED)
    mask_3 = 1 - mask_2
    pixels1 = mask_3[mask_3 > 0]
    n_pixels1 = pixels1.shape[0]
    L_f = np.sum(img_gray*mask_3)/n_pixels1
    displayIMG(mask_2*255,"Mask")
    pixels2 = mask_3[mask_3 == 0]
    n_pixels2 = pixels2.shape[0]
    other = img_gray*mask_2
    other_L = np.sum(img_gray*mask_2)/n_pixels2
    p_n = np.sum(mask_3)
    flat_sort = np.sort(other.flatten())
    other_L_n0 = flat_sort[p_n::]
    # other_L_n0 = other[other != 0]
    std_other_L = pow((np.sum(pow((other_L_n0-other_L),2))/n_pixels2),0.5)

print(area_f)    
print(radius_f)
print(L_f)
print(other_L)  
print(std_other_L) 
