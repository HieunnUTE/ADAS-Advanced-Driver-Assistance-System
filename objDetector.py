from ulti import *
from collections import deque, OrderedDict

class objDetector(YOLO):
    def __init__(self, model: str | Path = 'weight/obstacle.pt', task='detect', dict_setup = {},
                focal_length:float=3.67, sensor_height:float = 8.45,#8.45
                react_time= 0.5, deceleration= 5):
        
        super().__init__(model, task)
        
        self.dict_setup = dict_setup

        self.Height = { # mm
            0: 1700,    # Person
            1: 1800,    # Rider
            2: 1700,    # Car
            3: 3000,    # Truck
            4: 2800,    # Bus
            # 5: 2500,    # Train
            # 6: 1300,    # Moto
            # 7: 1300,    # Bicycle
        }
        
        # BDD100k
        self.class_label = ['Person', 'Rider', 'Car', 'Truck', 'Bus', 
                            'Train', 'Moto', 'Bicycle', 'Tf_light', 'Tf_sign']
        
        self.focal_length = focal_length    # (mm)
        self.sensor_height = sensor_height  # (mm)
        
        self.react_time = react_time    # (second)
        self.deceleration = deceleration    # (<0) (m/s^2)
                    
        self.currentObjInfo = []
        self.lastObjInfo = []
        

        self.currentSignInfo = []


        self.frame_id = 0
        self.dictVector = OrderedDict(maxlen=10)

        self.outputInfo:list = [None, None, None, "No_Object"]
    
    # Return self.objBoxes_list
    def objTrack(self, source=None, stream=False, persist=False,   **kwargs) -> list:
        self.result = super().track(source, stream, persist,  **kwargs)
        self.objBoxes_list = self.result[0].boxes.to('cpu').numpy()
        return self.objBoxes_list
    
    # Return apx_distance = round(apx_dis, 2)
    def calDistance(self, h, class_id:int) -> float:
        real_obj_height = self.Height[class_id]

        numerator = self.focal_length * real_obj_height * self.dict_setup['shape'][0]
        denominator = h * self.sensor_height
        
        apx_dis = (numerator/denominator)/1000 # --> Meter
        
        return round(apx_dis, 1)
    
    # Return safe_dis = round(safe_dis, 2)
    def calSafeDistance(self) -> float:
        """
        react_time: 0.5s - 500ms
        deceleration: 15 ft/s2 = 4.572 m/s2
        """
        stop_dis = (self.velocity**2)/(2*self.deceleration)
        react_dis = self.velocity*self.react_time         
        safe_dis = stop_dis + react_dis
        return round(safe_dis, 1)
    
    # Return: True/Flase
    def _isinPoly(self, xp, yp) -> bool:
        inside = False
        n = len(self.dict_setup['polygon'])
        p1x, p1y = self.dict_setup['polygon'][0]
        for i in range(n+1):
            p2x, p2y = self.dict_setup['polygon'][i % n]
            if yp > min(p1y, p2y):
                if yp <= max(p1y, p2y):
                    if xp <= max(p1x, p2x):
                        if p1y != p2y:
                            x = (yp - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or xp <= x:
                                inside = not inside
            p1x, p1y = p2x, p2y
        
        if not inside:
            if self.dict_setup['polygon'][1][0] <= xp <= self.dict_setup['polygon'][2][0]:
                inside = not inside
        return inside

    def calTTC(self, distance) -> float:
        """
        # Return: time_collision
        #   + If isinstance(float): round(time_collision, 2)
        #   + else:  None
        """
        if distance <= (self.velocity**2/9):
            delta = self.velocity**2 - 9*distance
            time_collision = round((self.velocity - np.sqrt(delta))/4.5, 1)

        else:
            time_collision = None
        
        return time_collision
    
    def collisionLogic(self, time_collision):
        if time_collision is not None:
            if 0.8 < time_collision <= 2.5:
                color = (255, 255, 0)
                collision_state = 'Attention: Forward Obstruction'
            elif 0.5 < time_collision <= 0.8:
                color = (0, 255, 255)
                collision_state = 'Attention: Imminent Collision'
            elif time_collision <= 0.5:
                color = (0, 0, 255)
                collision_state = "Warning: Approching Too Fast"
            else:
                color = (0, 255, 0)
                collision_state = "Message: Safe Distance"
        else:
            color = (0, 255, 0)
            collision_state = "Message: Safe Distance"
        
        return collision_state, color
    
    def processResult(self):
        self.currentObjInfo = []
        self.currentSignInfo = []
        
        if len(self.objBoxes_list) != 0:

            for box in self.objBoxes_list:
                class_id = box.cls[0]
                track_id = int(box.id[0]) if box.id is not None else -1

                if class_id in self.Height:
                    x, y, w, h = [int(cord_object) for cord_object in box.xywh[0]]
                    apx_distance = self.calDistance(h, class_id)
                    isPoly = self._isinPoly(x, y + h//2)
                    class_name = self.class_label[int(class_id)]
                    self.currentObjInfo.append([track_id, (x, y + h//2), apx_distance, isPoly, class_name])
                
                # if class_id == 9:
                #     if track_id != -1:
                #         signBox = [int(cord_traffic) if cord_traffic > 0 else 0 for cord_traffic in box.xyxy[0]]

                #         if len(signBox) !=0:
                #             w = signBox[2] - signBox[0]
                #             h = signBox[3] - signBox[1]
                            
                #             if (40,40) <= (w,h) <= (200, 200):
                #                 self.currentSignInfo.append(signBox)
                # else: pass
        if len(self.currentObjInfo) != 0:
            self.currentObjInfo = sorted(self.currentObjInfo, key=lambda s: (not s[3], s[2]))


        return self.currentObjInfo, self.currentSignInfo
    
    def getClosestObject(self) -> list:
        self.closestObjectInfo = [None, None, "No Forward Object", None] 
        if len(self.currentObjInfo) !=0:
            closest_object = self.currentObjInfo[0]

            if closest_object[3]:
                time_collision = self.calTTC(closest_object[2])
                collision_state, color = self.collisionLogic(time_collision)
                if closest_object[2] < 1:
                    # time_collision = 0.2
                    collision_state = 'Warning: Distance Too Close'
                    color = (0, 0, 255)
                # class_name, time_collision, collision_state, distance, color
                self.closestObjectInfo = [closest_object[-1], time_collision, collision_state, closest_object[2], color]
                
        return self.closestObjectInfo


    def getResult(self, frame, velocity:int=0):
        self.frame_id +=1
        self.velocity = velocity/3.6
        
        self.objBoxes_list = self.objTrack(frame, tracker='bytetrack.yaml', persist=True)
        
        self.safeDistance = self.calSafeDistance()

        self.currentObjInfo, self.currentSignInfo = self.processResult()
        self.closestObjectInfo = self.getClosestObject()
        self.outputInfo = self.closestObjectInfo[:3].copy()
        
        return self.outputInfo

    def updateVector(self):
        self.frame_id +=1
        
        if self.frame_id % 5 == 0:
            if len(self.currentObjInfo) != 0 and len(self.lastObjInfo) != 0:
                lastObjInfo_arr = np.array(self.lastObjInfo)
                for cur_obj in self.currentObjInfo:
                    track_id = cur_obj[0]
                    cur_pnt = cur_obj[1]
                    if track_id in lastObjInfo_arr[:, 0]:
                        index = np.where(lastObjInfo_arr[:, 0] == track_id)[0][0]
                        last_pnt = lastObjInfo_arr[index, 1]

                        # if min(cur_pnt[1], last_pnt[1]) == last_pnt[1] \
                        if track_id not in self.dictVector:
                            self.dictVector[track_id] = deque([cur_pnt], maxlen=10)
                        else:
                            self.dictVector[track_id].append(cur_pnt)

            self.lastObjInfo = self.currentObjInfo

        else: pass

    def drawResult(self, annotation_frame):
        
        cv2.putText(annotation_frame, f'Safe Distance: {self.safeDistance}', 
                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(annotation_frame, f'Object: {self.closestObjectInfo[0]}', 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(annotation_frame, f'Time Collision: {self.closestObjectInfo[1]}', 
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(annotation_frame, f'{self.closestObjectInfo[2]}',
                    (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.closestObjectInfo[-1], 2, cv2.LINE_AA)

        if len(self.currentObjInfo) != 0:
            for id, pos, d, isPoly, cls_name in self.currentObjInfo:
                if d == self.closestObjectInfo[-2]:
                    color = self.closestObjectInfo[-1]
                else:
                    color = (255, 0, 0) if isPoly else (255, 255, 255)
                
                # if id in self.dictVector:
                #     if len(self.dictVector[id]) > 1:
                #         for i in range(len(self.dictVector[id]) - 1):
                #             x1, y1 = self.dictVector[id][i]
                #             x2, y2 = self.dictVector[id][i+1]
                #             cv2.line(annotation_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotation_frame, pos, 2, color, -1)
                cv2.putText(annotation_frame, f'{cls_name}:{d}', 
                        (pos[0] - 10, pos[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


        return annotation_frame