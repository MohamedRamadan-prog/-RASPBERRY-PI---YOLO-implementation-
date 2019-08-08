from moviepy.editor import VideoFileClip
from yolo_pipeline import *


def pipeline_yolo(img):
    #img_undist = lane_process(img)
    output = vehicle_detection_yolo(img)
    return output

cap = cv2.VideoCapture(0)
#cap =  cv2.VideoCapture('project_video.mp4')
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (250,250))
ret,img = cap.read()

while True: 
    last_time = time.time()
    ret,img = cap.read()
    frame=cv2.resize(img,(250,250), interpolation = cv2.INTER_AREA)
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
            yolo_result = pipeline_yolo(frame)
            #print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            cv2.imshow('window',yolo_result)
            #out.write(yolo_result_resize)
    else:
        break 
cap.release()          
out.release()
cv2.destroyAllWindows()        
"""
out_file = 'outx.mp4'
clip = VideoFileClip('project_video.mp4')
clip_out= clip.fl_image(pipeline_yolo)
clip_out.write_videofile(out_file, audio=False)
"""