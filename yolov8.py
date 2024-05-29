import cv2
from ultralytics import YOLO
import numpy as np


# Load the YOLOv8 model  
model = YOLO(r"yolov8_detect.pt") 


# Open the video file
# video_path = r"D:\Downloads\beads_video.mp4"

video_path = 0
cap = cv2.VideoCapture(video_path)


def mapping(mid_x,mid_y):
    add_1x = 100
    add_1y = 200
    add_2x = 700
    add_2y = 800
    out_x = add_1x + (mid_x-0)/(600-0)*(add_2x-add_1x)
    out_y = add_1y + (mid_y-0)/(600-0)*(add_2y-add_1y)
    return [out_x,out_y]

# Loop through the video frames
while cap.isOpened():
    midpoints = []
    main_points = []
    # Read a frame from the video
    success, frame = cap.read()
    frame = cv2.resize(frame,(600,600))

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        frame_plot = results[0].plot()
        # cv2.putText(frame_plot,"Count => "  + str(len(results[0].boxes.cls.tolist())),(50,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2) 
        for i in results[0].boxes.xyxy:
            mid_x = float((i[0]+i[2])/2)
            mid_y = float((i[1]+i[3]/2))
            midpoints.append([mid_x,mid_y])
            main_points.append(mapping(mid_x,mid_y))
        print(main_points)

        # Display the annotated frame
        
        # final_result = np.concatenate([frame,frame_plot],axis=1)

        cv2.imshow("YOLOv8 Inference", frame_plot)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()






