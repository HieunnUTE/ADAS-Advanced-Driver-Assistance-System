# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:10:19 2021

@author: DROWSINESS DETECTION-"""

import mediapipe as mp
import cv2
from scipy.spatial import distance as dis

class drowsiDetector():
    def __init__(self) -> None:
        self.face_mesh = mp.solutions.face_mesh
        self.draw_utils = mp.solutions.drawing_utils
        self.landmark_style = self.draw_utils.DrawingSpec((0,255,0), thickness=2, circle_radius=2)
        self.connection_style = self.draw_utils.DrawingSpec((0,0,255), thickness=2, circle_radius=2)

        self.STATIC_IMAGE = False
        self.MAX_NO_FACES = 2
        self.DETECTION_CONFIDENCE = 0.6
        self.TRACKING_CONFIDENCE = 0.5

        self.COLOR_RED = (0,0,255)
        self.COLOR_BLUE = (255,0,0)
        self.COLOR_GREEN = (0,255,0)

        self.LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
                    185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]

        self.RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
        self.LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]


        self.LEFT_EYE_TOP_BOTTOM = [386, 374]
        self.LEFT_EYE_LEFT_RIGHT = [263, 362]

        self.RIGHT_EYE_TOP_BOTTOM = [159, 145]
        self.RIGHT_EYE_LEFT_RIGHT = [133, 33]

        self.UPPER_LOWER_LIPS = [13, 14]
        self.LEFT_RIGHT_LIPS = [78, 308]


        self.FACE=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
                    377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

        self.face_model = self.face_mesh.FaceMesh(static_image_mode=self.STATIC_IMAGE,
                                max_num_faces= self.MAX_NO_FACES,
                                min_detection_confidence= self.DETECTION_CONFIDENCE,
                                min_tracking_confidence= self.TRACKING_CONFIDENCE)
        
        self.frame_count = 0
        self.min_frame = 6
        self.min_tolerance = 5.0
        self.message = 'AWAKE'
        self.ratio_eyes = 3.2
        
    def euclidean_distance(self, image, top, bottom):
        height, width = image.shape[0:2]
                
        point1 = int(top.x * width), int(top.y * height)
        point2 = int(bottom.x * width), int(bottom.y * height)
        
        distance = dis.euclidean(point1, point2)
        return distance

    def get_aspect_ratio(self, image, outputs, top_bottom, left_right):
        landmark = outputs.multi_face_landmarks[0]
                
        top = landmark.landmark[top_bottom[0]]
        bottom = landmark.landmark[top_bottom[1]]
        
        top_bottom_dis = self.euclidean_distance(image, top, bottom)
        
        left = landmark.landmark[left_right[0]]
        right = landmark.landmark[left_right[1]]
        
        left_right_dis = self.euclidean_distance(image, left, right)
        
        aspect_ratio = left_right_dis/(top_bottom_dis+0.0001)
        
        return aspect_ratio
    
    def draw_landmarks(self, image, outputs, land_mark, color):
        height, width = image.shape[:2]
                
        for face in land_mark:
            point = outputs.multi_face_landmarks[0].landmark[face]
            
            point_scale = ((int)(point.x * width), (int)(point.y*height))
            
            cv2.circle(image, point_scale, 2, color, 1)
    
    def getState(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        outputs = self.face_model.process(image_rgb)
        if outputs.multi_face_landmarks:
           
            self.draw_landmarks(image, outputs, self.FACE, self.COLOR_GREEN)
            self.draw_landmarks(image, outputs, self.LEFT_EYE_TOP_BOTTOM, self.COLOR_RED)
            self.draw_landmarks(image, outputs, self.LEFT_EYE_LEFT_RIGHT, self.COLOR_RED)

            self.draw_landmarks(image, outputs, self.RIGHT_EYE_TOP_BOTTOM, self.COLOR_RED)
            self.draw_landmarks(image, outputs, self.RIGHT_EYE_LEFT_RIGHT, self.COLOR_RED)

            ratio_left =  round(self.get_aspect_ratio(image, outputs, self.LEFT_EYE_TOP_BOTTOM, self.LEFT_EYE_LEFT_RIGHT), 2)
            ratio_right =  round(self.get_aspect_ratio(image, outputs, self.RIGHT_EYE_TOP_BOTTOM, self.RIGHT_EYE_LEFT_RIGHT), 2)
            self.ratio_eyes = round((ratio_left + ratio_right)/2.0, 2)

            if self.ratio_eyes >= 3.5 or self.ratio_eyes <= 2.9:
                self.frame_count +=1
            else:
                self.frame_count = 0
                self.message = 'AWAKE'
                
            if self.frame_count > 15:
                self.message = 'SLEEP'

        else:
            self.message = 'FOCUS'
        
        cv2.putText(image, f'STATE: {self.message}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0,0,255) if self.message == 'SLEEP' or self.message == 'TIRED' else (0, 255, 0), 2, cv2.LINE_AA)
        return image, self.message
