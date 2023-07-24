from pathlib import Path
from typing import Union
import cv2
import numpy as np

from ultralytics import YOLO
import mysql.connector
import paramiko

trafficNames = ['GioiHanTocDo5','GioiHanTocDo10','GioiHanTocDo20','GioiHanTocDo30','GioiHanTocDo40','GioiHanTocDo50','GioiHanTocDo60','GioiHanTocDo70','GioiHanTocDo80','GioiHanTocDo90','GioiHanTocDo100','GioiHanTocDo110','GioiHanTocDo120','HetGioiHanTocDo','BoHetMoiLenhCam','CamXCVuot','ChoPhepXCVuot','VaoKhuDC','HetKhuDC','TocDoToiThieu30','TocDoToiThieu40','TocDoToiThieu60','TocDoToiThieu80','HetTocDoToiThieu','mandatory','regularity','warning']
velocity = 50
signal = False
sign_name = None

def setup(source:str):
    
    if source is None: 
        raise "Missing input source dir"

    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 20)


    ret, frame = cap.read()


    if not ret:
        raise "Something went wrong with Video Capture"
    
    
    height, width = frame.shape[:2]
    center = (width//2, height//2)
    
    polygonA = (center[0] - width//4 - 20, height - 10)
    polygonB = (width//2 - 40, (2*height//3 - 30))
    polygonC = (width//2 + 40, (2*height//3 - 30))
    polygonD = (center[0] + width//4 + 20, height - 10)
    polygon = [polygonA, polygonB, polygonC, polygonD]
    dict_setup = {
        'shape' : (height, width),
        
        'center': center,
        
        'laneThresh' : (center[0] - width//5 + 10, center[0] + width//5 - 10),
        
        'polygon': polygon,
        'pts_draw': [np.array([polygon], np.int32)]

    }

    return cap, dict_setup

def getFPS(prev_time):
    cur_time = cv2.getTickCount()
    time_diff = (cur_time - prev_time)/cv2.getTickFrequency()
    fps = 1/time_diff

    prev_time = cur_time

    return fps, prev_time

def videoWriter(shape, video_name='video_output', output_dir='output_video', fps=25):
    size =  (shape[1], shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_dir = f'{output_dir}/{video_name}.avi'
    videoOutput = cv2.VideoWriter(video_dir, fourcc, fps, size)
    return videoOutput

class SQLConnector:
    def __init__(self, host=('localhost'), user=('hieu'), password=('123'), database=('assistance_car')):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        self.cursor = None

    def connect(self):
        self.conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        self.cursor = self.conn.cursor()

    def execute_query(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def execute_write(self, query):
        self.cursor.execute(query)
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()