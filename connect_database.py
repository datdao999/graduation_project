import sqlite3
import datetime

from numpy.core.records import record

def connect ():
    try:
        sqliteConnection = sqlite3.connect('car.db')
        cursor = sqliteConnection.cursor()
        print("Database created and Successfully Connected to SQLite")
        
    except sqlite3.Error as error:
        print(error)
def insert(license_plate):
    try:
        sqliteConnection = sqlite3.connect('car.db')
        cursor = sqliteConnection.cursor()
        time = datetime.datetime.now()
        conn = cursor.execute('insert into carIn(license,time) values (?,?);', (license_plate, time))
        sqliteConnection.commit()
        print('them thanh cong', conn.rowcount)
        cursor.close()
    except sqlite3.Error as err:
        print("Loi them vao database",err)
def select_car_in(license_plate):
    try:
        sqliteConnection = sqlite3.connect('car.db')
        cursor = sqliteConnection.cursor()
        cursor.execute('select * from carIn where license like ?', [license_plate])
        record = cursor.fetchone()
        cursor.close()
        return record
        
    except sqlite3.Error as err:
        print("Loi them vao database",err)

def select_car_out(license_plate):
    try:
        sqliteConnection = sqlite3.connect('car.db')
        cursor = sqliteConnection.cursor()
        cursor.execute('select * from carOut where licensePlate like ?', [license_plate])
        record = cursor.fetchall()
        cursor.close()
        return record
    except sqlite3.Error as err:
        print('Loi:', err)


def delete(license_plate):
    try:
        sqliteConnection = sqlite3.connect('car.db')
        cursor = sqliteConnection.cursor()
        conn = cursor.execute('delete from carIn where license like ?',[license_plate])
        sqliteConnection.commit()
        print('delete thanh cong', conn.rowcount)
        cursor.close()
    except sqlite3.Error as err:
        print("Loi delete  database",err)

def insert_car_out (license_plate, price, time_in):
    try:
        sqliteConnection = sqlite3.connect('car.db')
        cursor = sqliteConnection.cursor()
        time_out = datetime.datetime.now()
        cursor.execute('insert into carOut(licensePlate, price, timeIn, timeOut) values(?,?,?,?)', [license_plate, price, time_in, time_out])
        sqliteConnection.commit()
        cursor.close()
    except sqlite3.Error as err:
        print('Loi them vao', err)

def select_price():
    try:
        sqliteConnection = sqlite3.connect('car.db')
        cursor = sqliteConnection.cursor()
        cursor.execute('select price, max(id) from pricePerHour ')
        record = cursor.fetchone()
        cursor.close()
        
        return record[0]
    except sqlite3.Error as err:
        print('loi khi lay gia', err)

def insert_price(price):
    try:
        sqliteConnection = sqlite3.connect('car.db')
        cursor = sqliteConnection.cursor()
        time = datetime.datetime.now()
        conn = cursor.execute('insert into pricePerHour(price, time) values(?,?)', [int(price), time])
        sqliteConnection.commit()
        print('them thanh cong', conn.rowcount)
        cursor.close()
    except  sqlite3.Error as err:
        print('loi khi them gia moi', err)

def check_sign_in (username):
    try:
        sqliteConnection = sqlite3.connect('car.db')
        cursor = sqliteConnection.cursor()
        cursor.execute('select username, password from signIn where username = ?',[username])
        record = cursor.fetchone()
        cursor.close()
        return record
    except sqlite3.Error as err:
        print(err)

# print(check_sign_in('fucckdmin'))
