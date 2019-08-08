from moviepy.editor import VideoFileClip
from yolo_pipeline import *

def pipeline_yolo(img):
    #img_undist = lane_process(img)
    output = vehicle_detection_yolo(img)
    return output

if __name__ == "__main__":
    
    demo = 1
    # 1:image (YOLO), 2: video (YOLO Pipeline)
    if demo == 1:
        filename = 'examples/test46.jpg'
        image = mpimg.imread(filename)

        #(1) Yolo pipeline
        yolo_result = pipeline_yolo(image)
        plt.figure()
        plt.imshow(yolo_result)
        plt.title('yolo pipeline', fontsize=30)

    elif demo == 2:
        # YOLO Pipeline
        video_output = 'examples/project_YOLO_OUT3.mp4'
        clip1 = VideoFileClip("examples/project_video.mp4")#.subclip(30,32)
        clip = clip1.fl_image(pipeline_yolo)
        clip.write_videofile(video_output, audio=False)
