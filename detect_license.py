import cv2
import numpy as np 
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob
import functools
def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def getPlate(image_path, Dmax = 608, Dmin = 256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2]))/ min(vehicle.shape[:2])
    side = int (ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ ,LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor
test_image = 'test/test5.jpg'
LpImg,cor = getPlate(test_image)
print("Detect %i plate(s) in"%len(LpImg), splitext(basename(test_image))[0])
print("Coordiate of plates in image: \n", cor)
cv2.imshow("bien so xe", LpImg[0])
if (len(LpImg)):
    # plate_img = cv2.convertScaleAbs(LpImg[0], alpha = (255.0))
    # roi_gray = cv2.cvtColor(plate_img,cv2.COLOR_BGR2GRAY)
    # roi_blur = cv2.GaussianBlur(roi_gray,(3,3),1)
    # ret,thre = cv2.threshold(roi_blur,120,255,cv2.THRESH_BINARY_INV)
    # kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # thre_mor = cv2.morphologyEx(thre,cv2.MORPH_DILATE,kerel3)
    # cont,_ = cv2.findContours(thre_mor,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # areas_ind = {}
    # areas = []
    # for ind,cnt in enumerate(cont) :
    #     area = cv2.contourArea(cnt)
    #     areas_ind[area] = ind
    #     areas.append(area)
    # areas = sorted(areas,reverse=True)[2:9]
    # print("day la thong tin ve areas"+str(areas))
    # for i in areas:
    #     (x,y,w,h) = cv2.boundingRect(cont[areas_ind[i]])
    #     cv2.rectangle(plate_img,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow('ket qua', plate_img)

    plate_img = cv2.convertScaleAbs(LpImg[0], alpha = (255.0))
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Show the original image
    cv2.imshow("License Plate", plate_img)

    # Apply Gaussian blurring and thresholding 
    # to reveal the characters on the license plate
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
    cv2.imshow('anh nhi phan', thresh)
    # Perform connected components analysis on the thresholded images and
    # initialize the mask to hold only the components we are interested in
    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")
    def imshow_components(labels):
    # Map component labels to hue val
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0

        cv2.imshow('labeled.png', labeled_img)
        
    imshow_components(labels)

    # Set lower bound and upper bound criteria for characters
    total_pixels = plate_img.shape[0] * plate_img.shape[1]
    lower = total_pixels // 70 # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 20 # heuristic param, can be fine tuned if necessary

    # Loop over the unique components
    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue
    
        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
    
        # If the number of pixels in the component is between lower bound and upper bound, 
        # add it to our mask
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    # Find contours and get bounding box for each contour
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # Sort the bounding boxes from left to right, top to bottom
    # sort by Y first, and then sort by X if Ys are similar
    def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]
    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )
i = 0
model_svm = cv2.ml.SVM_load('svm.xml')
for bound in boundingBoxes:
    print("in lan thu {}", i)
    (x, y, w, h) = bound
    cv2.rectangle(plate_img, (x, y), (x+w, y+h), (0,255,0),2)
    i = i+1
    result = model_svm.predict(bound)
    print(bound)
cv2.imshow('ket qua', plate_img)

print("boundingBoxes:" + str(boundingBoxes) )


#     plate_img = cv2.convertScaleAbs(LpImg[0], alpha = (255.0))
#     gray = cv2.cvtColor(plate_img,cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (7,7), 0)
#     binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
#     kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

# cv2.imshow('anh binary', binary)
# def sort_contour(cnts, reverse = False):
#    i =0
#    boundingBoxes = [cv2.boundingRect(c) for c in  cnts] 
#    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
#    return cnts
# cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# test_roi = plate_img.copy()
# crop_characters = []
# digit_w, digit_h = 30, 60
# for c in sort_contour(cont):
#     (x, y, w, h) = cv2.boundingRect(c)
#     ratio = h/w
#     if 1 <= ratio <=3.5:
#         if  h/plate_img.shape[0] >=0.5:
#             cv2.rectangle(test_roi, (x,y), (x+w, y+h), (0,255,0),2)
#             curr_num = thre_mor[y:y+h, x:x+w]
#             curr_num = cv2.resize(curr_num, dsize = (digit_w, digit_h))
#             _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#             crop_characters.append(curr_num)

# print("phat hien {} ky tu".format(len(crop_characters)))

# #cv2.imshow('anh ky tu', crop_characters[7])

cv2.waitKey(0)
