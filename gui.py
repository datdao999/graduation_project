from datetime import datetime
from sqlite3.dbapi2 import Date
import cv2
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox
from detect_license import load_model, preprocess_image, getPlate, imshow_components, compare, recognize
import functools
from os.path import splitext, basename
import numpy as np 
import connect_database
import count_money
import hashlib


def sign_in():
    def check(username, password):
        hash = hashlib.md5(password.encode())
        information = connect_database.check_sign_in(username) 
        if information == None:
            messagebox.showerror("showerror", "You type wrong username or password") 
        elif username == information[0] and hash.hexdigest() == information[1]:
            fifth_root.destroy()
            manager()
    fifth_root = Tk()
    fifth_root.title('Sign in')
    fifth_root.geometry('350x200')
    photo_home = PhotoImage(file = 'arrow-return-down-left-icon.png')
    label = Label(fifth_root, text='Log in', font=("Arial", 24), fg='blue')
    label.place(x=120, y = 0)
    user_name_label = Label(fifth_root, text='Username')
    user_name_label.place(x=0, y = 70)
    user_name_entry = Entry(fifth_root)
    user_name_entry.place(x = 60, y =70)
    password_label = Label(fifth_root, text='Password')
    password_label.place(x=0, y = 100)
    password_entry = Entry(fifth_root, show='*')
    password_entry.place(x=60, y=100)
    sign_in_button = Button(fifth_root, text='Submit', command= lambda:check(user_name_entry.get(), password_entry.get()))
    sign_in_button.place(x = 60, y = 130)
    home_btn = Button(fifth_root, text= 'Home', image=photo_home, command= lambda: [fifth_root.destroy(),start_page()], compound=LEFT)
    home_btn.place(x = 0, y = 170)

    fifth_root.mainloop()


def start_page():
    root = Tk()
    root.title('Recognize license plate')
    root.geometry('300x300')
    root.resizable(width= False, height= False) 
    photo_about_me = PhotoImage(file = 'Actions-help-about-icon.png')
    photo_car = PhotoImage(file ='Car-icon.png')
    photo_manage_price = PhotoImage(file = 'Manager-icon.png')
    photo_logo = ImageTk.PhotoImage(Image.open('logo_utc2.jfif').resize((150,150)))

    logo_label = Label(root, image= photo_logo)
    logo_label.place(x = 70, y = 0, height = 150, width = 150)

    recognize_btn = Button(root, text= 'Recognize license', image=photo_car, command= lambda:[root.destroy(), call_recognize()], compound=LEFT)
    recognize_btn.place(x = 70, y = 160)

    manager_btn = Button(root, text= 'Manager price', image = photo_manage_price ,command=lambda:[root.destroy(), sign_in()], compound=LEFT)
    manager_btn.place(x = 70, y = 200)

    about_me_btn = Button(root, text='About me', image= photo_about_me ,command= lambda:[root.destroy(), open_about_me()] , compound= LEFT )
    about_me_btn.place(x = 70, y = 240)

    

    root.mainloop() 

def open_about_me():
    second_root = Tk()
    second_root.title('About me')
    second_root.geometry('400x300')
    photo_home = PhotoImage(file = 'arrow-return-down-left-icon.png')
    photo_logo = ImageTk.PhotoImage(Image.open('logo_utc2.jfif').resize((150,150)))

    logo_label = Label(second_root, image= photo_logo)
    logo_label.place(x = 130, y = 0, height = 150, width = 150)
    introduc_string = 'Develop by Dao Duc Dat\n MSV:585107120 - CNTTK58'
   
    about_me_label = Label(second_root, text = introduc_string)
    about_me_label.place(x= 130, y=180)
    
    open_root_button = Button(second_root, text= 'Home', image = photo_home ,command= lambda:[second_root.destroy(), start_page()], compound=LEFT)
    open_root_button.place(x = 0, y=270)
    second_root.mainloop()
    
def call_recognize():
    my_image2 = ''
    my_image = ''
    third_root = Tk()
    third_root.title('Reconize license plate')
    third_root.geometry('900x600')
    third_root.resizable(width= False, height= False)
    in_label= Label(third_root, text= 'Entrance', font=("Arial", 24), fg='blue')
    in_label.place(x=120, y = 0)
    photo_home = PhotoImage(file = 'arrow-return-down-left-icon.png')
    photo_load_image = PhotoImage(file = 'My-Pictures-icon.png')
    photo_execute = PhotoImage(file = 'execute-icon.png')

    wpod_net_path = "wpod-net-upgrade_final.json"
    wpod_net = load_model(wpod_net_path)

    def opendiglog() :
        global my_image, filename
        filename = filedialog.askopenfilename( title = 'select a file', filetypes = (('jpg files', '*.jpg'),('png files', '*.png'),('all files', '*.*' )))
        my_image = ImageTk.PhotoImage(Image.open(filename).resize((350,250)))

        # label =  Label(root, image= my_image)
        label.config(image=my_image) 
        # label.place(x=100, y=0)
        
    def excute(filename):

        license_string = recognize(filename, wpod_net= wpod_net) 

        text.delete('1.0', END)
        text.insert(INSERT, license_string)
        print("Ky tu cua xe la", license_string)
    
        connect_database.insert(license_string)
    

        # cv2.waitKey()
        
    label =  Label(third_root, bg='gray', image= my_image)
    label.place(x=0, y=40, width=350, height=250)

    buttonLoadImage = Button(third_root, text='Load_image', image= photo_load_image ,command= opendiglog, compound= LEFT)
    buttonLoadImage.place(x=180, y=330)

    buttonExcute = Button(third_root, text='Excute', image=photo_execute ,command=lambda:excute(filename), compound=LEFT)
    buttonExcute.place(x=300, y=330,)

    license_label_in = Label(third_root, text = 'License plates:' )
    license_label_in.place(x= 0, y = 300)

    text = Text(third_root)
    text.place (x=0, y=330, height = 30, width = 150)

    my_canvas = Canvas(third_root, width=5, height= 600, bg='white')
    my_canvas.place(x=450, y = 0)
    my_canvas.create_line(2, 0, 2, 600, fill='black',dash=(4,2))
    #Exit gateway

    def opendiglog2() :
        global my_image2, filename2
        filename2 = filedialog.askopenfilename( title = 'select a file', filetypes = (('jpg files', '*.jpg'),('png files', '*.png'),('all files', '*.*' )))
        my_image2 = ImageTk.PhotoImage(Image.open(filename2).resize((350,250)))
        # label2 =  Label(root, image= my_image)
        label2.config(image=my_image2)
        # label2.place(x=500, y=0)

    def excute2(filename):

        license_string = recognize(filename, wpod_net= wpod_net) 


        try:
            time_getout = connect_database.select_car_in(license_string)[2]
            connect_database.delete(license_string)
            text2.delete('1.0', END)
            text2.insert(INSERT, license_string)
            print("Ky tu cua xe la", license_string)
        
            time = datetime.now() - datetime.strptime(time_getout, '%Y-%m-%d %H:%M:%S.%f') 
            # print('Time: ', time)
            print('thanh tien la:',count_money.count(time))
            connect_database.insert_car_out(license_string,count_money.count(time), time_getout )
            text3.delete('1.0', END)
            text3.insert(INSERT, str(count_money.count(time))) 
        except:
            text2.delete('1.0', END)
            text2.insert(INSERT, 'the car is not in the park')
        

    out_label= Label(third_root, text= 'Exit', font=("Arial", 24), fg='blue')
    out_label.place(x=650, y = 0)


    label2 =  Label(third_root, bg='gray', image= my_image2)
    label2.place(x=549, y=40, width=350, height=250)

    buttonLoadImage2 = Button(third_root, text='Load_image', image=photo_load_image ,command=opendiglog2, compound= LEFT)
    buttonLoadImage2.place(x=715, y=330)


    buttonExcute2 = Button(third_root, text='Excute', image= photo_execute ,command=lambda:excute2(filename2), compound=LEFT)
    buttonExcute2.place(x=830, y=330)

    license_label_out = Label(third_root, text = 'License plates:' )
    license_label_out.place(x= 530, y = 300)

    text2 = Text(third_root)
    text2.place (x=530, y=330, height = 30, width = 150)

    price_label = Label(third_root, text = 'Fee:' )
    price_label.place(x= 530, y = 370)

    text3 = Text(third_root)
    text3.place(x=530, y=400, height = 30, width = 150)
    text3.bind("<Key>", lambda e: 'break')

    home_btn = Button(third_root, text= 'Home', image=photo_home, command= lambda: [third_root.destroy(),start_page()], compound=LEFT)
    home_btn.place(x = 0, y = 570)

    third_root.mainloop()

def manager():

    fouth_root = Tk()
    fouth_root.geometry('500x400')
    fouth_root.title('Manager')
    photo_home = PhotoImage(file = 'arrow-return-down-left-icon.png')
    def check_input(S):
        if S in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return True
        fouth_root.bell() # .bell() plays that ding sound telling you there was invalid input
        return False

    def manager_price(price):
        if price != "":
            connect_database.insert_price(price)
            label.config(text='Price:'+str(connect_database.select_price()))
        else:
            messagebox.showerror("showerror", "You need input the price")

    def load_history(license_plate):
        rows = connect_database.select_car_out(license_plate)
        for row in rows:
            my_tree.insert(parent='', index='end', values=(row[1], row[2], row[3], row[4]))
        

    price_label = Label(fouth_root, text='New price:')
    price_label.place(x = 0, y = 0)
    vcmd = (fouth_root.register(check_input), '%S')
    price_entry = Entry(fouth_root, validate= 'key', vcmd= vcmd)
    price_entry.place(x = 70, y = 0)
    price_button = Button(fouth_root, text= 'Apply', command=lambda:manager_price(price_entry.get()))
    price_button.place(x=69, y=30)
    label = Label(fouth_root, text = 'Price:'+str(connect_database.select_price()))
    label.place(x = 250, y = 0)

    search_label = Label(fouth_root, text='Search:')
    search_label.place(x = 0, y = 75)
    
    search_entry = Entry(fouth_root)
    search_entry.place(x = 70, y =75 )

    search_button = Button(fouth_root, text='Search', command=lambda:load_history(search_entry.get()))
    search_button.place(x = 70, y= 105)


    my_tree = ttk.Treeview(fouth_root)

    my_tree['columns'] = ('license plate', 'price', 'time in', 'time out')
    my_tree.column('#0', width=0, minwidth=0)
    my_tree.column('license plate', width= 90 )
    my_tree.column('price', anchor=W, width=90  )
    my_tree.column('time in',anchor=CENTER, width=90 )
    my_tree.column('time out', anchor=W , width=90 )

    my_tree.heading('license plate', text='Licensen plate')
    my_tree.heading('price', text='Price')
    my_tree.heading('time in', text='Time in')
    my_tree.heading('time out', text='Time out')
    
    my_tree.place(y = 150, x = 0)
    home_btn = Button(fouth_root, text= 'Home', image=photo_home, command= lambda: [fouth_root.destroy(),start_page()], compound=LEFT)
    home_btn.place(x = 430, y = 370)

    fouth_root.mainloop()

start_page()



