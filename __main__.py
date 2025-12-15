import cv2
import time
import platform
import numpy as np
import mediapipe as mp
from pynput.mouse import Button, Controller

from nonmouse.args import *
from nonmouse.utils import *

mouse = Controller()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    cap_device, mode, kando, screenRes = tk_arg()
    dis = 0.7                          
    preX, preY = 0, 0
    nowCli, preCli = 0, 0            
    norCli, prrCli = 0, 0             
    douCli = 0                         
    i, k, h = 0, 0, 0
    LiTx, LiTy, list0x, list0y, list1x, list1y, list4x, list4y, list6x, list6y, list8x, list8y, list12x, list12y, list20x, list20y = [
    ], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []   # 移動平均用リスト
    moving_average = [[0] * 3 for _ in range(3)]
    nowUgo = 1
    cap_width = 1280
    cap_height = 720
    start, c_start = float('inf'), float('inf')
    c_text = 0
    scroll_down_sensitivity = -3       # Sensitivity for scrolling down
    scroll_up_sensitivity = 3          # Sensitivity for scrolling up

    window_name = 'NonMouse'
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cfps = int(cap.get(cv2.CAP_PROP_FPS))
    if cfps < 30:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
        cfps = int(cap.get(cv2.CAP_PROP_FPS))
   
    ran = max(int(cfps/10), 1)
    hands = mp_hands.Hands(
        min_detection_confidence=0.8,   # 
        min_tracking_confidence=0.8,    #
        max_num_hands=1                 # 
    )
    double_click_start = None
    double_click_detected = False

    # Load gestures
    gestures_folder = 'gestures'
    gestures = load_gestures(gestures_folder)

    
    while cap.isOpened():
        p_s = time.perf_counter()
        success, image = cap.read()
        if not success:
            continue

        if mode == 1:                   # Mouse
            image = cv2.flip(image, 0)  # 上下反転
        elif mode == 2:                 # Touch
            image = cv2.flip(image, 1)  # 左右反転

        
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False   
        results = hands.process(image)  
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
       
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # preX, preY
            if i == 0:
                preX = hand_landmarks.landmark[8].x
                preY = hand_landmarks.landmark[8].y
                i += 1

            landmark0 = [calculate_moving_average(hand_landmarks.landmark[0].x, ran, list0x), calculate_moving_average(
                hand_landmarks.landmark[0].y, ran, list0y)]
            landmark1 = [calculate_moving_average(hand_landmarks.landmark[1].x, ran, list1x), calculate_moving_average(
                hand_landmarks.landmark[1].y, ran, list1y)]
            landmark4 = [calculate_moving_average(hand_landmarks.landmark[4].x, ran, list4x), calculate_moving_average(
                hand_landmarks.landmark[4].y, ran, list4y)]
            landmark6 = [calculate_moving_average(hand_landmarks.landmark[6].x, ran, list6x), calculate_moving_average(
                hand_landmarks.landmark[6].y, ran, list6y)]
            landmark8 = [calculate_moving_average(hand_landmarks.landmark[8].x, ran, list8x), calculate_moving_average(
                hand_landmarks.landmark[8].y, ran, list8y)]
            landmark12 = [calculate_moving_average(hand_landmarks.landmark[12].x, ran, list12x), calculate_moving_average(
                hand_landmarks.landmark[12].y, ran, list12y)]
            landmark20 = [calculate_moving_average(hand_landmarks.landmark[20].x, ran, list20x), calculate_moving_average(
                hand_landmarks.landmark[20].y, ran, list20y)]

         
            absKij = calculate_distance(landmark0, landmark1)
            
            absUgo = calculate_distance(landmark8, landmark12) / absKij
            
            absCli = calculate_distance(landmark4, landmark6) / absKij

            posx, posy = mouse.position

            # 人差し指の先端をカーソルに対応
            # カメラ座標をマウス移動量に変換
            nowX = calculate_moving_average(
                hand_landmarks.landmark[8].x, ran, LiTx)
            nowY = calculate_moving_average(
                hand_landmarks.landmark[8].y, ran, LiTy)

            dx = kando * (nowX - preX) * image_width
            dy = kando * (nowY - preY) * image_height

            if platform.system() in ['Windows', 'Linux']:     # Windows,linux
                dx = dx+0.5
                dy = dy+0.5
            preX = nowX
            preY = nowY
            # print(dx, dy)
            if posx+dx < 0: 
                dx = -posx
            elif posx+dx > screenRes[0]:
                dx = screenRes[0]-posx
            if posy+dy < 0:
                dy = -posy
            elif posy+dy > screenRes[1]:
                dy = screenRes[1]-posy

            
            # click状態
            if absCli < dis:
                nowCli = 1          # nowCli:左クリック状態(1:click  0:non click)
                draw_circle(image, hand_landmarks.landmark[8].x * image_width,
                            hand_landmarks.landmark[8].y * image_height, 20, (0, 250, 250))
            elif absCli >= dis:
                nowCli = 0
            if np.abs(dx) > 7 and np.abs(dy) > 7:
                k = 0                           

          
            index_middle_distance = calculate_distance(landmark8, landmark12) / absKij
            if index_middle_distance < dis:
                norCli = 1
                draw_circle(image, hand_landmarks.landmark[8].x * image_width,
                            hand_landmarks.landmark[8].y * image_height, 20, (0, 0, 250))
            else:
                norCli = 0

            # Detect if last three fingers are raised
            last_three_fingers_raised = (
                hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y and  # Middle finger
                hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y and  # Ring finger
                hand_landmarks.landmark[20].y < hand_landmarks.landmark[19].y      # Pinky finger
            )

            # Detect if last two fingers are raised
            last_two_fingers_raised = (
                hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y and  # Ring finger
                hand_landmarks.landmark[20].y < hand_landmarks.landmark[19].y      # Pinky finger
            )

            # Handle gestures exclusively
            if last_three_fingers_raised:
                # Perform only page down action
                mouse.scroll(0, scroll_down_sensitivity)  # Scroll down
                # Draw an orange circle on the middle finger
                draw_circle(image, hand_landmarks.landmark[12].x * image_width,
                            hand_landmarks.landmark[12].y * image_height, 20, (0, 165, 255))  # Orange color
            elif last_two_fingers_raised:
                # Perform only page up action
                mouse.scroll(0, scroll_up_sensitivity)  # Scroll up
                # Draw a purple circle on the ring finger
                draw_circle(image, hand_landmarks.landmark[16].x * image_width,
                            hand_landmarks.landmark[16].y * image_height, 20, (128, 0, 128))  # Purple color
            else:
                # Perform other actions only if no scroll gestures are active
                if absUgo >= dis and nowUgo == 1:
                    mouse.move(dx, dy)
                    draw_circle(image, hand_landmarks.landmark[8].x * image_width,
                                hand_landmarks.landmark[8].y * image_height, 8, (250, 0, 0))
                if nowCli == 1 and nowCli != preCli:
                    if h == 1:                                
                        h = 0
                    elif h == 0:                              
                        mouse.press(Button.left)
                if nowCli == 0 and nowCli != preCli:
                    mouse.release(Button.left)
                    k = 0
                if norCli == 1 and norCli != prrCli:
                    mouse.press(Button.right)
                    mouse.release(Button.right)
                    h = 1

            preCli = nowCli
            prrCli = norCli

        # 表示 #################################################################################
        cv2.putText(image, "cameraFPS:"+str(cfps), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        p_e = time.perf_counter()
        fps = str(int(1/(float(p_e)-float(p_s))))
        cv2.putText(image, "FPS:"+fps, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        dst = cv2.resize(image, dsize=None, fx=0.4,
                         fy=0.4)         # HDの0.4倍で表示
        cv2.imshow(window_name, dst)
        if (cv2.waitKey(1) & 0xFF == 27) or (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 0):
            break
    cap.release()

if __name__ == "__main__":
    main()