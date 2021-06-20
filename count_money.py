from datetime import datetime, timedelta
import connect_database
def count(time):
    money = 0
    if time.days >= 1:
        money = money + time.days * 24 * connect_database.select_price()
    if int(time.seconds / 3600) >=1:
        money = money + int(time.seconds/3600) * connect_database.select_price()
    else: 
        money = money + connect_database.select_price()
    return money

