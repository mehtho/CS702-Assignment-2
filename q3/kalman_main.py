import cv2
from KalmanFilter import KalmanFilter
import pandas as pd
import numpy as np

def scale_coordinates(x, y, ori_size=(1920, 1080), down_size=(1280, 720)):
    scale_x = down_size[0] / ori_size[0]
    scale_y = down_size[1] / ori_size[1]
    new_x = int(x * scale_x)
    new_y = int(y * scale_y)
    return new_x, new_y

def main():
    VideoCap = cv2.VideoCapture('./q3/hand_tracking_output.MOV')
    df = pd.read_csv("./q3/txys_missingdata.csv")
    ControlSpeedVar = 50  #Lowest: 1 - Highest:100
    HiSpeed = 100

    KF = KalmanFilter(dt=0.1, std=0.1)

    old_loc = []
    while(True):
        loc = []
        # Read frame
        ret, frame = VideoCap.read()
     
        current_frame_index = (int(VideoCap.get(cv2.CAP_PROP_POS_FRAMES)) - 1) * 100
        if (current_frame_index < 2000):
            continue
        filtered_rows = df[df["t_ms"] == current_frame_index]

        if not filtered_rows.empty:
            loc.append(np.array([[int(filtered_rows.x_px)], [int(filtered_rows.y_px)]]))            
            old_loc = loc

        if (len(loc) > 0):  # Have data
            # Predict
            (x, y) = KF.predict()
            # Update
            (x1, y1) = KF.update(loc[0])
            update_x, update_y = scale_coordinates(x1, y1)
        else:  # Missing data
            (x, y) = KF.predict()
            (x1, y1) = KF.update(np.array([x, y]).reshape(2, 1))  # Use predicted value to update
            update_x, update_y = scale_coordinates(x1, y1)
            # Draw a circle as the estimated object position
            cv2.circle(frame, (int(update_x), int(update_y)), 15, (0, 0, 255), 2)

            old_loc = []
            old_loc.append(np.array([[int(x1)], [int(y1)]]))

        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)


if __name__ == "__main__":
    main()
