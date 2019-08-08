from moviepy.editor import VideoFileClip
from yolo_pipeline import *
import cv2
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (250,250))

def pipeline_yolo(img):
    #img_undist = lane_process(img)
    output = vehicle_detection_yolo(img)
    return output

def screen_record(): 
    #last_time = time.time()
    while(True):
        # 800x600 windowed mode
        ret, frame = cap.read()
        #printscreen =  np.array(ImageGrab.grab(bbox=(0,0,800,600)))
        #RGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yolo_result = pipeline_yolo(frame)
        #print('loop took {} seconds'.format(time.time()-last_time))
        #last_time = time.time()
        cv2.imshow('window',yolo_result)
        #out.write(yolo_result)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            #out.release()
            cv2.destroyAllWindows()
            break
if __name__ == "__main__":
    screen_record()
