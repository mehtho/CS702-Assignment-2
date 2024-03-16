import numpy as np
import pandas as pd
import cv2

# Load fingertip tracking data
data = pd.read_csv("./q3/txys_missingdata.csv")

# Load filtered trajectory points
filtered_trajectory_df = pd.read_csv('filtered_trajectory.csv')
filtered_trajectory = [(int(row['x']), int(row['y'])) for index, row in filtered_trajectory_df.iterrows()]

# Visualize trajectory overlaid on video
cap = cv2.VideoCapture("./q3/hand_tracking_output.mov")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('./q3/kalman_filter_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    for point in filtered_trajectory:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
    
    out.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
