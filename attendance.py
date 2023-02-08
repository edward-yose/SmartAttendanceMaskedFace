import datetime

import cv2
from datetime import datetime as dt
from datetime import date
import os
import numpy as np

path = r'../imageDatabase/'
directory_attendance = r"./AttendanceData/"


def create_simple_file_attendance():
    get_date = date.today().strftime("%Y_%m_%d")
    get_path = directory_attendance + get_date + ".csv"
    if not os.path.exists(get_path):
        with open(get_path, 'w+') as f:
            f.write("Name, Room ID, Datetime\n")
            f.close()
            print("[INFO]", get_date + ".csv has been created in", directory_attendance, "Folder")
    return str(get_path)


def create_complex_file_attendance():
    get_date = date.today().strftime("%Y_%m_%d")
    get_path = directory_attendance + get_date + ".csv"
    if not os.path.exists(get_path):
        with open(get_path, 'w+') as f:
            f.write("Name, Room, Entry Time, Minutes, Last Update Time \n")
            f.close()
            print("[INFO]", get_date + ".csv (Complex) has been created in", directory_attendance, "Folder")
    return str(get_path)


class AttendanceEntry:
    def mark_attendance(name):
        if not os.path.exists("attendance.csv"):
            with open("attendance.csv", 'w+') as f:
                f.write("Name, Datetime\n")
                f.close()
        with open('attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = dt.now()
                dtStr = now.strftime('%Y-%m-%d %H:%M:%S')
                f.writelines(f'\n{name}, {dtStr}')
                print("[INFO] Name ", name, " checked for attendance")
                f.close()


class DailyAttendanceEntry:
    def mark_this_day_attendance(name, room_id):
        get_path = create_simple_file_attendance()
        with open(get_path, 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            roomList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList and name != "":
                now = dt.now()
                dtStr = now.strftime('%Y-%m-%d %H:%M:%S')
                f.writelines(f'\n{name}, {room_id}, {dtStr}')
                print("[INFO] Name ", name, " checked for attendance in room", room_id)
                f.close()


# Create API Based Attendance System to Databases - Not Finished
class AttendanceEntryDB:
    def __init__(self):
        pass

    def mark_attendance(name, room):
        print("Check")

