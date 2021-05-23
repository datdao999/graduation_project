import cv2
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from detect_license import load_model, preprocess_image, getPlate, imshow_components, compare
import functools
from os.path import splitext, basename
import numpy as np 

root = Tk()
root.title('Reconize license plate')
root.geometry('800x600')
root.resizable(width= False, height= False)
frame = Frame(root,  bg = 'green')
frame.place(x=0, y=0)
image_dir = 'D:/Project graduate/license_plate_WPOD/test/thuty.png'
global text
def opendiglog() :
    global filename
    filename = filedialog.askopenfilename( title = 'select a file', filetypes = (('png files', '*.png'),('jpg files', '*.jpg'),('all files', '*.*' )))
    my_image = ImageTk.PhotoImage(Image.open(filename).resize((300,300)))
    label =  Label(root, image= my_image)
    label.image = my_image 
    label.place(x=100, y=0)
   
    
def recognize(filename):
    wpod_net_path = "wpod-net.json"
    wpod_net = load_model(wpod_net_path)


    test_image = filename
    # cv2.imshow('dsfdsf', test_image)
    LpImg,cor = getPlate(test_image, wpod_net)
    print("Detect %i plate(s) in"%len(LpImg), splitext(basename(test_image))[0])
    print("Coordiate of plates in image: \n", len(LpImg))
    
    if (len(LpImg)):
        

        plate_img = cv2.convertScaleAbs(LpImg[0], alpha = (255.0))
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # Show the original image
        

        # Apply Gaussian blurring and thresholding 
        # to reveal the characters on the license plate
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel3)
        #cv2.imshow('anh nhi phan', thre_mor)
        # Perform connected components analysis on the thresholded images and
        # initialize the mask to hold only the components we are interested in
        _, labels = cv2.connectedComponents(thresh)
        mask = np.zeros(thresh.shape, dtype="uint8")
        
            
        imshow_components(labels)

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
        
        # print("ratio2:", ratio)
        if ratio <= 9.0 and ratio > 1.1:
            
            cv2.rectangle(plate_img, (x, y), (x+w, y+h), (0,255,0),2)
            output_string =str(y) 
            cv2.putText(plate_img, str(output_string), (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
            curr_num = thre_mor[ y:y+h, x:x+w]
            curr_num = cv2.resize(curr_num, dsize = (30, 60))

            _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
            curr_num = np.array(curr_num, dtype = np.float32)
            curr_num = curr_num.reshape(-1, 30*60)
            i = i+1
            
            result = model_svm.predict(curr_num)[1]
            string = model_svm.getKernelType()
            print("kieu thuat toan", string)
            result = int(result[0,0])
            if result<=9: # Neu la so thi hien thi luon
                result = str(result)
            else: #Neu la chu thi chuyen bang ASCII
                result = chr(result)
            
            license_list.extend(result)
        license_string = ''.join(license for license in license_list)
    text.insert(INSERT, license_string)
    print("Ky tu cua xe la", license_string)
    cv2.imshow('ket qua', plate_img)
   
    # print("boundingBoxes:" + str(boundingBoxes) )


    cv2.waitKey()
    

# dialog = filedialog.askopenfile(title = 'select a file', filetypes = (('png files', '*.png'),('all files', '*.*' )))


buttonLoadImage = Button(root, text='Load_image', command=  opendiglog)
buttonLoadImage.place(x=0, y=0)

label =  Label(root, bg='gray')
label.place(x=100, y=0, width=300, height=200)

root.update()
print("label width: ", label.winfo_geometry())

buttonExcute = Button(root, text='Excute', command=lambda:recognize(filename))
buttonExcute.place(x=0, y=40,)


text = Text(root)
text.place (x=0, y=80, height = 100, width = 100)


root.mainloop() 
