from moviepy.editor import VideoFileClip
from yolo_pipeline import *


def pipeline_yolo(img):
    #img_undist = lane_process(img)
    output = vehicle_detection_yolo(img)
    return output

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output11.avi', fourcc, 20.0, (640,480))
#cap =  cv2.VideoCapture('project_video.mp4')




while( cap.isOpened() ) : 
    last_time = time.time()
    ret,img = cap.read()
    #img=cv2.resize(frame,(250,250), interpolation = cv2.INTER_AREA) 
    # 800x600 windowed mode
    #printscreen =  np.array(ImageGrab.grab(bbox=(0,0,800,600)))
    #printscreen = cv2.VideoCapture(0)
    #RGB=cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)
    #yolo_result = pipeline_yolo(RGB)
    if ret == True:    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        else:
            #print(img.shape)
            yolo_result = pipeline_yolo(img)
            #print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            cv2.imshow('window',yolo_result)
            out.write(yolo_result)
    else:
        break 
#cap.release()          
out.release()
cv2.destroyAllWindows()        
"""
out_file = 'outx.mp4'
clip = VideoFileClip('project_video.mp4')
clip_out= clip.fl_image(pipeline_yolo)
clip_out.write_videofile(out_file, audio=False)
"""