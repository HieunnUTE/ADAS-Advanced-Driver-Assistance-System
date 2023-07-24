from laneDetector import laneDetector
from objDetector import objDetector
from signDetector import signDetector
from drowsiDetector import drowsiDetector
from ulti import *

video_path = 'output_video/video_real_1.avi'
txt_path = video_path.replace('.avi', '.txt')

cap1, dict_setup = setup(source=video_path)
cap2 = cv2.VideoCapture(0)
# cap3 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

prev_time = cv2.getTickCount()

objDetector = objDetector(dict_setup = dict_setup)
laneDetector = laneDetector(dict_setup = dict_setup)
signDetector = signDetector()
drowsiDetector = drowsiDetector()
connector  = SQLConnector()

with open(txt_path, 'r') as f:
    # Đọc các dòng từ tệp và lưu chúng thành một danh sách
    lines = f.readlines()

frame_id = 0
while cap1.isOpened():
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    # success3, frame3 = cap3.read()
    if success1:
        connector.connect()
        velocity = int(lines[frame_id].split(':')[1])
        # velocity = 50
        frame_id +=1
        connector.execute_write(f"UPDATE speed SET speed = '{velocity}';")

        laneState = laneDetector.laneLogic(frame1, velocity, signal)
        laneFrame = laneDetector.laneDraw(frame1, velocity = velocity, unit = 'kmh', signal_bool=signal)

        output_obj = objDetector.getResult(frame1, velocity)
        objFrame  = objDetector.drawResult(laneFrame)
        output_obj.append(laneState)

        sign_name, annotationFrame = signDetector.getSignName(frame1, objFrame)
        driv_frame, driv_state = drowsiDetector.getState(frame2)
        
        
        # UPDATE SQL
        connector.execute_write(f"UPDATE result SET object = '{output_obj[0]}', time_collision = '{output_obj[1]}', collision_state = '{output_obj[2]}', lane_state = '{output_obj[3]}';")
        connector.execute_write(f"UPDATE sign SET signname = '{sign_name}';")
        connector.execute_write(f"UPDATE driver SET driver_state = '{driv_state}';")
        
        fps, prev_time = getFPS(prev_time)
        print('*'*100)
        print(f'Frame id: {objDetector.frame_id} || FPS: {fps:.1f} || Signal: {signal}')
        print(f'Speed: {velocity} kmh || Lane_state: {output_obj[3]} || Driver state: {driv_state} || Sign: {sign_name}')
        print(f'Object: {output_obj[0]} || Distance: {objDetector.closestObjectInfo[-2]} m || Time Collision: {output_obj[1]} s || Collision State: {output_obj[2]}')
        
        # tf_frame = objDetector.result[0].plot()

        # cv2.imshow("Result LANE", lane_frame)
        cv2.imshow("RESULT", annotationFrame)
        # cv2.imshow("Result TRAFFIC SIGN", tf_frame)
        driv_frame = cv2.resize(driv_frame, (320, 240))

        
        cv2.imshow("DRIVER STATE", driv_frame)

        # display = cv2.resize(frame3, (320, 240))
        # cv2.imshow("DISPLAY - WARNING", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        elif cv2.waitKey(1) & 0xFF == ord("s"):
            signal = not signal
        
        connector.close()

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap1.release()
cap2.release()
cv2.destroyAllWindows()
