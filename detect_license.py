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
        model.load_weights('wpod-net-upgrade_final.h5')
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def getPlate(image_path, wpod_net, Dmax = 608, Dmin = 256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2]))/ min(vehicle.shape[:2])
    side = int (ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ ,LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor
def imshow_components(labels):
    # Map component labels to hue val
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0

        # cv2.imshow('labeled.png', labeled_img)
def compare(rect1, rect2):
        
        if abs(rect1[1] - rect2[1]) > 20:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]

def recognize(test_image, wpod_net):

    LpImg,cor = getPlate(test_image, wpod_net)
    print("Detect %i plate(s) in"%len(LpImg), splitext(basename(test_image))[0])
    print("Coordiate of plates in image: \n", cor)

    if (len(LpImg)):
        
        plate_img = cv2.convertScaleAbs(LpImg[0], alpha = (255.0))
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # Show the original image
        

        # Apply Gaussian blurring and thresholding 
        # to reveal the characters on the license plate
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
        # cv2.imshow("gaussian", blurred)
        # cv2.imshow('adaptiveThreshold', thresh)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel3)
        # cv2.imshow("morphologyEx", thre_mor)
    
        # Perform connected components analysis on the thresholded images and
        # initialize the mask to hold only the components we are interested in
        _, labels = cv2.connectedComponents(thresh)
        mask = np.zeros(thresh.shape, dtype="uint8")
        

        
        # imshow_components(labels)

        # Set lower bound and upper bound criteria for characters
        total_pixels = plate_img.shape[0] * plate_img.shape[1]
        lower = total_pixels // 190 # heuristic param, can be fine tuned if necessary
        upper = total_pixels // 20 # heuristic param, can be fine tuned if necessary
        # print("lower :{}, upper:{}".format(lower, upper))
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
            # print('numberpixels:',numPixels )
            # If the number of pixels in the component is between lower bound and upper bound, 
            # add it to our mask
            if numPixels > lower and numPixels < upper:
                
                mask = cv2.add(mask, labelMask)
            
        # Find contours and get bounding box for each contour
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]


        # Sort the bounding boxes from left to right, top to bottom
        # sort by Y first, and then sort by X if Ys are similar
        
        boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )

    i = 0
    model_svm = cv2.ml.SVM_load('svm.xml')
    license_list =[]
    for bound in boundingBoxes:
        (x, y, w, h) = bound
        ratio = h/w
        s = w * h
        white = cv2.countNonZero(thre_mor[ y:y+h, x:x+w])
        ratioWhite = white / s
        # print("w, h", w,h)
        
        # print('so white, dien tich ne:', ratioWhite)
        if  ratio <= 8.6 and ratio > 0.4 and ratioWhite > 0.35 and ratioWhite < 0.85 and w+x <= plate_img.shape[1] - 15 and x > 5:
            
            cv2.rectangle(plate_img, (x, y), (x+w, y+h), (0,255,0),2)
            
            output_string =str(x+w) 
            # cv2.putText(plate_img, str(output_string), (x-10, y+h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
            curr_num = thre_mor[ y:y+h, x:x+w]
            curr_num = cv2.resize(curr_num, dsize = (30, 60))
            
            _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
            curr_num = np.array(curr_num, dtype = np.float32)
            curr_num = curr_num.reshape(-1, 30*60)
            i = i+1
            result = model_svm.predict(curr_num)[1]
            string = model_svm.getKernelType()
            result = int(result[0,0])
            if result<=9: # Neu la so thi hien thi luon
                result = str(result)
            else: #Neu la chu thi chuyen bang ASCII
                result = chr(result)
            print("ky tu nhan dang: ", result)
            license_list.extend(result)
        license_string = ''.join(license for license in license_list)
    # cv2.imshow('ket qua', plate_img)
    # cv2.waitKey()
    return license_string

    


    
# wpod_net_path = "wpod-net-upgrade_final.json"
# wpod_net = load_model(wpod_net_path)
# recognize('../CarTGMT/AEONTP_6S81U5_checkin_2020-1-13-16-18bx9UOV6rY5.jpg', wpod_net= wpod_net) 
