from pathlib import Path
from typing import Union
from ulti import *

class signDetector(YOLO):
    def __init__(self, model: str | Path = 'weight/traffic_sign.pt', task='detect'):
        super().__init__(model, task)
        self.class_label = ['HetKhuDC', 'GioiHanTocDo50', 'GioiHanTocDo60','GioiHanTocDo70', 'GioiHanTocDo80', 'VaoKhuDC']
        self.sign_name = ''
    
    def signTrack(self, source=None, stream=False, persist=False,   **kwargs) -> list:
        self.result = super().track(source, stream, persist,  **kwargs)
        self.signBox_list = self.result[0].boxes.to('cpu').numpy()
        return self.signBox_list
    
    def getSignName(self, frame, anotation_frame):
        self.signBox_list = self.signTrack(frame, tracker='bytetrack.yaml', persist=True, conf=0.8, classes=[1, 2 ,3 ,4])
        if len(self.signBox_list) != 0:
            self.sign_name = ''
            for box in self.signBox_list:
                class_id = box.cls[0]
                xl, yl, xr, yr = [int(cord_object) for cord_object in box.xyxy[0]]
                class_name = self.class_label[int(class_id)]
                self.sign_name += class_name + ','
                cv2.rectangle(anotation_frame, (xl, yl), (xr, yr), (0, 255, 0), 2)
                cv2.putText(anotation_frame, class_name.replace('GioiHanTocDo',''), (xl,yl-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0,255,0), 2, cv2.LINE_AA)
        return self.sign_name, anotation_frame



            