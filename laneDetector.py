from ulti import *

class laneDetector(YOLO):
    
    def __init__(self, model: str | Path = 'weight/lane.pt', task='segment', dict_setup = {}):
        super().__init__(model, task)
        self.count = 0
        self.dict_setup = dict_setup
    
    def laneTrack(self, source=None, stream=False, persist=False,  **kwargs):
        
        self.result = super().track(source, stream, persist,  **kwargs)
        self.laneBoxes = self.result[0].boxes.xyxy.to('cpu').numpy()
        return self.laneBoxes
        
    def laneLogic(self, frame, velocity:int=0, signal_bool:bool=False):
        """
        Function used for lane logic

        input: 
            velocity:int (km/h) | default = 0
            signal_bool:boolean | default = Flase
        
        output: state:str
            "Lane-Missing"
            "Stop-detecting"
            "Good-keeping" | "Wrong-left" | "Wrong-right" | "Wrong-both"
        """
        if velocity < 30:
            self.state = "Stop-detecting"
        else:
            self.laneBoxes = self.laneTrack(frame, tracker='bytetrack.yaml')
            if len(self.laneBoxes) != 0:
                # Lấy vị trí của hai đường viền bên trái và bên phải
                x_tl, y_tl, x_br, y_br = self.laneBoxes[0]
                # self.center = (int((x_tl + x_br)/2), int((y_br + y_tl)/2))
                
                if (x_br - x_tl)//self.dict_setup['shape'][1] > 0.9 \
                        and (y_br - y_tl)//self.dict_setup['shape'][0] > 0.9: 
                    self.state = "Stop-Detecting"
                    return self.state
                
                # Lệch lane + signal_bool
                if x_tl < self.dict_setup['laneThresh'][0] and x_br > self.dict_setup['laneThresh'][1]:
                    # Nếu xe đang đi giữa hai làn đường đúng
                    self.state = "Good-keeping"
                    self.count = 0
                else:
                    if not signal_bool:
                        if self.count < 10:
                            self.count +=1
                            self.state = "Good-keeping"
                        
                        else:
                            # Nếu xe đang lệch khỏi làn đường
                            if self.dict_setup['laneThresh'][0] < x_tl and x_br < self.dict_setup['laneThresh'][1]:
                                self.state = "Wrong-both"
                            else:
                                if x_tl > self.dict_setup['laneThresh'][0]: 
                                    self.state = "Wrong-left"
                                elif x_br < self.dict_setup['laneThresh'][1]: 
                                    self.state = "Wrong-right"
                                else: pass
                    else:
                        self.state = "Changing-lane"
            else:
                self.state = "Lane-Missing"
        return self.state

    def laneDraw(self, frame, velocity, unit, signal_bool=False):
        
        annotation_frame = frame.copy()

        if int(velocity) > 30:
            annotation_frame = self.result[0].plot(line_width = 1, hide_conf=True)
            # for x,y in self.laneMasks:
            #     color = (255, 0, 0) if int(x) < self.dict_setup['center'][0] else (0, 0, 255)
            #     cv2.circle(annotation_frame, (int(x), int(y)), 5, color, -1)

        cv2.line(annotation_frame, (self.dict_setup['center'][0], self.dict_setup['shape'][0]-30),
                    (self.dict_setup['center'][0], self.dict_setup['shape'][0]-5), (255,0,255), 5)
        
        cv2.line(annotation_frame, (self.dict_setup['laneThresh'][0], self.dict_setup['shape'][0]-30),
                    (self.dict_setup['laneThresh'][0], self.dict_setup['shape'][0]-5), (255,255,255), 5)    
        
        cv2.line(annotation_frame, (self.dict_setup['laneThresh'][1], self.dict_setup['shape'][0]-30),
                    (self.dict_setup['laneThresh'][1], self.dict_setup['shape'][0]-5), (255,255,255), 5)

        cv2.polylines(annotation_frame, self.dict_setup['pts_draw'], True, (255, 255, 255), 2)
        
        cv2.putText(annotation_frame, f'Speed: {velocity} {unit}', (self.dict_setup['shape'][1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0,255,0), 2, cv2.LINE_AA)
        
        cv2.putText(annotation_frame, f"Signal: {'ON' if signal_bool else 'OFF'}", 
                        (self.dict_setup['shape'][1] - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0,255,0), 2, cv2.LINE_AA)
        
        if self.state:
            cv2.putText(annotation_frame, f'Lane state: {self.state}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0,255,0), 2, cv2.LINE_AA)
        
        return annotation_frame