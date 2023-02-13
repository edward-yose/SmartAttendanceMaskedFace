import datetime

import cv2
from datetime import datetime as dt
from datetime import date
import os
import numpy as np

path_fps = 'report_fps.csv'
path_para = 'report_para.csv'


def create_fps_file():
    if not os.path.exists(path_fps):
        with open('report_fps.csv', 'w+') as f:
            print("triggered")
            f.write("FPS, Time\n")
            f.close()
    return str(path_fps)


def create_para_file():
    if not os.path.exists(path_para):
        with open('report_para.csv', 'w+') as f:
            f.write("Time, Status\n")
            f.close()
    return str(path_para)


class GenerateReport:
    def generate_fps_report(fps: str):
        file = create_fps_file()
        with open(file, 'a') as f:
            gettime = datetime.datetime.now().strftime("%H:%M:%S.%f")
            # print(fps, gettime)
            f.writelines(f'\n{fps}, {gettime}')
            f.close()

    def generate_para_report(status: str):
        file = create_para_file()
        with open(file, 'a') as f:
            gettime = datetime.datetime.now().strftime("%H:%M:%S.%f")
            f.writelines(f'\n{gettime}, {status}')
            print(gettime, status)
            f.close()
