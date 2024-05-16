import mediapipe as mp
import numpy as np
import cv2
import pyautogui
from pynput import keyboard
from pynput.keyboard import Controller, Key
import one_euro_filter

import collections
import time
import os
import re

import alignment
import util
import sys
# sys.path.insert(0, './emnist_model/')

import torch
import mnist_model.pytorch_model_class
import emnist_model.pytorch_model_class
from emnist_model.pytorch_model_class import DEVICE

import keyboard_util

from sys import platform

# For PyInstaller Internal folder
bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))

def load_temp():
    confident_col = np.ones((21,1))
    templates = []
    templates_category = []
    # files = ["temp_data/arrow_temp.csv", "temp_data/fist_temp.csv","temp_data/five_temp.csv","temp_data/four_temp.csv","temp_data/one_temp.csv",]
    file_path = os.path.join(bundle_dir, 'data/template_image_new_wide_data/')
    files = os.listdir(file_path)
    print(files)
    for file in files:
        name = re.findall(r'(\w+).', file)[0]
        temp = np.hstack((np.loadtxt(file_path + file, dtype = float, delimiter=','), confident_col))
        templates.append(temp)
        templates_category.append(name)
    return templates, templates_category


def flip_hand(handedness):
    if handedness == 'Right':
        return 'Left'
    else:
        return 'Right'

# organize landmarks for cursor control hand tracking
def hand_keypoints(hand_landmarks):
    points = []
    for landmark in hand_landmarks.landmark:
        points.append([landmark.x, landmark.y, landmark.z])
    return np.array(points)

# https://en.wikipedia.org/wiki/Shoelace_formula to determine area
def shoelace_area(points):
    x0, y0 = np.hsplit(points, 2)
    points1 = np.roll(points, -1, axis=0)
    x1, y1 = np.hsplit(points1, 2)
    combination = x0 * y1 - x1 * y0
    area = np.sum(combination) / 2
    return area, x0 + x1, y0 + y1, combination

def palm_center(keypoints):
    indices = [0, 1, 5, 9, 13, 17]   # indices of mediapipe landmarks
    points = keypoints[indices, :2]  # x and y coordinates of landmarks
    area, x, y, combination = shoelace_area(points)
    center_x = np.sum(x * combination) / (6 * area)
    center_y = np.sum(y * combination) / (6 * area)
    center = np.array([center_x, center_y])
    radius = int(np.min(np.linalg.norm(points - center, axis=1)) * np.mean(image.shape[:2]))
    center = tuple(np.int32(center * image.shape[1::-1]))
    return center, radius

# def absolute(center):
#     scale = 2
#     print("center:", center)
#     x = center[0] * screen_width // (scale * image_width)
#     y = center[1] * screen_height // (scale * image_height)
#     print(x, y)
#     pyautogui.moveTo(x, y, _pause=False)

def absolute_scale(hand_center, move_func = lambda x, y : pyautogui.moveTo(x, y, _pause=False)):
    top_left = [0.50 * image_width, 0.50 * image_height]
    scale = screen_width // (0.33 * image_width)
    #scale_y = screen_height // (0.5 * height)
    out_x = scale * (hand_center[0] - top_left[0])
    out_y = scale * (hand_center[1] - top_left[1])
    move_func(out_x, out_y)


# def joystick(center, frame):
#     mouse_vector = center - joystick_center
#     length = np.linalg.norm(mouse_vector)
#     if length > joystick_radius:
#         mouse_vector = mouse_vector - np.array([joystick_radius, joystick_radius])
#         # mouse_vector = mouse_vector / length * (length - joystick_radius)
#         # mouse_move = np.multiply(np.power(abs(mouse_vector), 1.75) * 0.05, np.sign(mouse_vector))
#         pyautogui.move(np.int32(mouse_vector)[0], np.int32(mouse_vector)[1], _pause=False)
#         print('mouse vector', mouse_vector)
#     cv2.line(frame, tuple(joystick_center), tuple(np.int32(center)), (255, 0, 0), 2)
    
def get_filtered_center(center, t):
    x_center = x_filter(t, center[0])
    y_center = y_filter(t, center[1])
    return [x_center, y_center]


# mediapipe setup
mp_drawing, mp_drawing_styles = util.mediapipe_draw_setup()
hands, mp_hands = util.mediapipe_hand_setup()

# gesture recognition setup
templates, templates_category = util.load_temp(os.path.join(bundle_dir, 'data/template_image_new_wide_data/'))

# Webcam resolution:
#   https://stackoverflow.com/questions/19448078/python-opencv-access-webcam-maximum-resolution
# video streaming setup
cap = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# print(f"Attempted to set frame width, currently {str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# check system type
system_type = platform
if system_type.lower() == "darwin":
    system_type = "Mac"
elif system_type.lower().startswith("win"):
    system_type = "Windows"
else:
    system_type = "Linux"

# -------- begin keyboard input setup --------
keyboard = Controller()
is_keyboard_mode = True # TODO: will need to modify

# set up left and right list for gesture averaging
right_gesture_list = []
left_gesture_list = []

# array to hold drawn points
fingertip_path_right = []
new_fingertip_segment = []

# drag hyperparameters
min_drag = 10 # must drag at least 5 pixels to register as drag
drag_flag = False # RUSHING or DRAGGING? - Whiplash (2014)
drag_button = None

# 3D touch flag
holding_five = False

# store previous gesture for rising edge of gesture
prev_left_gesture = None
prev_right_gesture = None
prev_center = None
backspace_available = False
space_available = False
right_gesture = None
left_gesture = None
right_landmarks = None
left_landmarks = None
left_rising_edge_gesture = False
right_rising_edge_gesture = False
new_fingertip_segment_appended = False

drawn_image = None

# load ML classification model
digit_model = mnist_model.pytorch_model_class.NetReluShallow(D_in=28 * 28, H1=100, H2=100, D_out=10)
digit_model.load_model(path = os.path.join(bundle_dir, './mnist_model/saved_models'))

char_model = emnist_model.pytorch_model_class.CNN_SRM().to(DEVICE)
char_model.load_model(path = os.path.join(bundle_dir, './emnist_model/saved_models'))

# mapping from model outputs to digits / characters
# digit mapping based on mnist dataset
digit_map = {
    0: '0',  1: '1',  2: '2',  3: '3',  4: '4',  5: '5',  6: '6',  7: '7',  8: '8', 9: '9'
}

# mapping of characters to digits
# mapping based on https://arxiv.org/pdf/1702.05373.pdf, balanced dataset+++
char_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# we use char recognition by default
input_mode = 1 # 1 for char, -1 for digit

#variables to keep track of confidence selection
choice1 = None
choice2 = None
choice_made = None
choice1_start_percentage = None
choice1_end_percentage = None
choice2_start_percentage = None
choice2_end_percentage = None

# -------- end keyboard input setup --------


# mode_mapping = {1: 'cursor', 2: 'scroll', 3: 'volume', 4: 'window', 5: 'safari'}
mode_mapping = {"fist_left": 'eye', "fist": 'eye', "one_left": 'cursor', "two_left": 'scroll', "three_left": 'volume', "four_left": 'window', "five_left": 'browser'}

time_start = None
leftclick_start = None
rightclick_start = None
# center_queue = collections.deque(5 * [(0, 0)], 5)

success, image = cap.read()
screen_width, screen_height = pyautogui.size()
image_height, image_width = image.shape[:2]
# joystick_center = np.array([int(0.75 * image_width), int(0.5 * image_height)])
# joystick_radius = 40

# mouse filtering setup
x_filter = one_euro_filter.OneEuroFilter(0, 0.675 * image_width)
y_filter = one_euro_filter.OneEuroFilter(0, 0.65 * image_height)
frame = 0

pyautogui.FAILSAFE = False

win_name = "Mouse/Keyboard Controller"
# cv2.namedWindow(win_name, cv2.WND_PROP_ASPECT_RATIO)
cv2.imshow(win_name, image)
cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)

while cap.isOpened():
    # get tick count for measuring latency
    # https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html
    tick_start = cv2.getTickCount()

    success, image = cap.read()
    image = cv2.flip(image, 1)

    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # always show bounding boxes if in keyboard mode
    if is_keyboard_mode:
        if choice1 != None and choice2 != None:
            image, choice1_start_percentage, \
            choice1_end_percentage, \
            choice2_start_percentage, choice2_end_percentage = \
                keyboard_util.confidence_selection(image, choice1, choice2)
        else:
            # draw the bounding box for the backspace area of the screen
            image, bksp_space_end_percentage = keyboard_util.draw_modifiers_boxes(image)

    # get hand keypoints
    mp_success, num_hands, results = util.mediapipe_process(image, hands)
    if mp_success:
        for i in range(num_hands):
            score, handedness, hand_landmarks = util.get_mediapipe_result(results, i)

            hand_gesture = util.recognize_gesture(templates, templates_category, hand_landmarks)
            util.mediapipe_draw(image, hand_landmarks, mp_hands, mp_drawing, mp_drawing_styles)

            landmark_data = []
            for point in hand_landmarks.landmark:
                landmark_data.append([point.x, point.y])
            
            if handedness == 'Right':
                right_gesture = keyboard_util.avg_gesture(right_gesture_list, hand_gesture)
                right_landmarks = hand_landmarks
                cv2.putText(image, 'right hand gesture: ' + str(right_gesture),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
            else:
                left_gesture = keyboard_util.avg_gesture(left_gesture_list, hand_gesture)
                left_landmarks = hand_landmarks
                cv2.putText(image, 'left hand gesture: ' + str(left_gesture),
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

            # update rising edge for updating previous gestures
            if prev_right_gesture != right_gesture:
                right_rising_edge_gesture = True
            if prev_left_gesture != left_gesture:
                left_rising_edge_gesture = True

        if left_gesture == "fist_left" and right_gesture == "fist" \
           and (prev_left_gesture != "fist_left" or prev_right_gesture != "fist"):
            is_keyboard_mode = not is_keyboard_mode

# ------------------ BEGIN MOUSE MODE ------------------
        if not is_keyboard_mode:
            if left_gesture in mode_mapping:
                mode = mode_mapping[left_gesture]
                cv2.putText(image, 'mode: ' + mode, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
                if mode == 'scroll':
                    if right_gesture == "one":
                        # pyautogui.press('up')
                        pyautogui.scroll(10)
                    elif right_gesture == 'arrow':
                        pyautogui.press('down')
                        # pyautogui.press('down')
                        pyautogui.scroll(-10)
                    elif right_gesture == "two":
                        pyautogui.hscroll(50)
                    elif right_gesture == "three":
                        pyautogui.hscroll(-50) 
                elif mode == 'eye':
                    output = face_mesh.process(image)
                    landmark_points = output.multi_face_landmarks
                    frame_h, frame_w, _ = image.shape
                    if landmark_points:
                        landmarks = landmark_points[0].landmark

                        # Right eye ball landmarks (pupil boundaries)
                        right_eyeball = range(474, 478)
                        for id in right_eyeball:
                            landmark = landmarks[id]
                            x = int(landmark.x * frame_w)
                            y = int(landmark.y * frame_h)
                            cv2.circle(image, (x, y), 3, (0, 255, 0)) # Right eye green
                        
                        # Calculate eye center
                        eye_anchor = [263, 463, 374, 386]

                        eye_anchor_x = 0
                        eye_anchor_y = 0

                        for id in eye_anchor:
                            landmark = landmarks[id]
                            x = int(landmark.x * frame_w)
                            y = int(landmark.y * frame_h)

                            eye_anchor_x += int(x / len(eye_anchor))
                            eye_anchor_y += int(y / len(eye_anchor))
                            cv2.circle(image, (x, y), 3, (255, 0, 255)) # Anchor Purple
                        cv2.circle(image, (eye_anchor_x, eye_anchor_y), 3, (255, 0, 0)) # Anchor Blue

                        # -------------------------------
                        # Left eye tracking for blinking
                        # -------------------------------
                        left = [landmarks[145], landmarks[159]]
                        for landmark in left:
                            x = int(landmark.x * frame_w)
                            y = int(landmark.y * frame_h)
                            cv2.circle(image, (x, y), 3, (0, 255, 255)) # Yellow left eye lids
                        if (left[0].y - left[1].y) < 0.004:
                            pyautogui.click()
                        
                        # -------------------------------
                        # Create vector between pupil and anchor
                        # -------------------------------
                        pupil_x, pupil_y = 0, 0
                        for id in right_eyeball:
                            landmark = landmarks[id]
                            pupil_x += landmark.x
                            pupil_y += landmark.y
                        
                        pupil_x /= len(right_eyeball)
                        pupil_y /= len(right_eyeball)
                        pupil_center_x = int(pupil_x * frame_w)
                        pupil_center_y = int(pupil_y * frame_h)

                        # pupil center
                        cv2.circle(image, (pupil_center_x, pupil_center_y), 3, (255, 0, 0), -1) # Right Pupil Blue

                        # vector
                        cv2.line(image, (eye_anchor_x, eye_anchor_y), (pupil_center_x, pupil_center_y), (0, 0, 255), 2) # Pupil Anchor Diff red

                elif mode == 'cursor' and right_landmarks is not None:
                    frame += 1 # used for filter
                    keypoints = hand_keypoints(right_landmarks)
                    center, radius = palm_center(keypoints)
                    center = get_filtered_center(center, frame)
                    # center_queue.appendleft(center)
                    # center = np.mean(center_queue, axis=0)
                    #absolute(center)
                    absolute_scale(center)
                    # scale = screen_width // (0.5 * image_width)
                    #joystick(center, image)

                    # center and palm tracking
                    cv2.circle(image, tuple(np.int32(center)), 2, (0, 255, 0), 2)
                    cv2.circle(image, tuple(np.int32(center)), radius, (0, 255, 0), 2)
                    # cv2.circle(image, tuple(joystick_center), joystick_radius, (255, 0, 0), 2)

                    # hand movement area
                    # cv2.line(image, (0.5 * width, 0.25 * image_height), (0.5 * width, 0.75 * image_height), (0, 255, 0), 3)
                    # cv2.line(image, (1 * width, 0.25 * image_height), (1 * width, 0.75 * image_height), (0, 255, 0), 3)
                    # cv2.line(image, (0.5 * width, 0.25 * image_height), (1 * width, 0.25 * image_height), (0, 255, 0), 3)
                    # cv2.line(image, (0.5 * width, 0.75 * image_height), (1 * width, 0.75 * image_height), (0, 255, 0), 3)
                    cv2.rectangle(image, (int(0.50 * image_width), int(0.50 * image_height)), (int(0.85 * image_width), int(0.80 * image_height)), (0, 255, 0), 3)
                    
                    # Begin/close 3d touch
                    if right_gesture == 'five' and not holding_five and system_type == "Mac":
                        pyautogui.click()
                        keyboard.press(Key.space)
                        holding_five = True
                        
                    # Close 3d touch
                    elif holding_five and right_gesture != 'five' and system_type == "Mac":
                        keyboard.release(Key.space)
                        holding_five = False

                    elif right_gesture == 'arrow' and prev_right_gesture != 'arrow' and not drag_flag:
                        pyautogui.click()
                        prev_center = center
                        
                    elif right_gesture == "two" and prev_right_gesture != "two" and not drag_flag:
                        pyautogui.doubleClick()
                    
                    elif right_gesture == "one" and prev_right_gesture != "one" and not drag_flag:
                        pyautogui.rightClick()
                        prev_center = center
                    
                    # support for right click drag
                    elif right_gesture == 'one' and prev_right_gesture == 'one' and not drag_flag:
                        diff = (center[0] - prev_center[0], center[1] - prev_center[1])
                        if max(*map(abs, diff)) > min_drag:
                            drag_flag = True
                            drag_button = 'right'
                            
                    # support for left click drag
                    elif right_gesture == 'arrow' and prev_right_gesture == 'arrow' and not drag_flag:
                        diff = (center[0] - prev_center[0], center[1] - prev_center[1])
                        if max(*map(abs, diff)) > min_drag:
                            drag_flag = True
                            drag_button = 'left'
                            
                    elif right_gesture == 'two' and drag_flag:
                        drag_flag = False
                        absolute_scale(prev_center)
                        pyautogui.click() if drag_button == 'left' else pyautogui.rightClick()
                        absolute_scale(center, move_func = lambda x, y : pyautogui.dragTo(x, y, button = drag_button))
                        prev_center = None
                        
                    # --- note: the commented code below allows for "continuous" clicking ---
                    #     where holding the gesture allows for clicking every second
                    #     at the cost of extreme camera lag. This may be wanted, but is
                    #     commented out for now
                    # if right_gesture == 'arrow' and prev_right_gesture != 'arrow':
                        # if not leftclick_start:
                            # leftclick_start = time.time()
                        # elif time.time() - leftclick_start <= 1:
                            # continue
                        # else:
                            # leftclick_start = None
                            # pyautogui.click()
                    # -------- end note -----------
                        
                elif mode == 'volume':
                    if time_start is None:
                        time_start = time.time()
                    time_stamp = time.time()
                    if (time_stamp - time_start > 1):
                        time_start = time_stamp
                        if right_gesture == "one":
                            keyboard.tap(Key.media_volume_up)
                        elif right_gesture == 'arrow':
                            keyboard.tap(Key.media_volume_down)
                        elif right_gesture == "two":
                            keyboard.tap(Key.media_volume_mute)
                elif mode == 'window':
                    # if time_start is None:
                    #     time_start = time.time()
                    # time_stamp = time.time()
                    # if (time_stamp - time_start > 1):
                    #     time_start = time_stamp
                    if right_gesture == "one" and prev_right_gesture != "one": #switch to previous app
                        # pyautogui.hotkey('command', 'tab')
                        # added Windows shortcut commands
                        if system_type == "Mac":
                            pyautogui.hotkey('command', 'tab')
                        elif system_type == "Windows":
                            pyautogui.hotkey('alt', 'tab')
                        # elif right_gesture == "two": #browse windows
                        #     # pyautogui.hotkey('ctrl', 'up')
                        #     if system_type == "Mac":
                        #         pyautogui.hotkey('ctrl', 'up')
                        #     elif system_type == "Windows":
                        #         pyautogui.hotkey('ctrl', 'alt', 'tab')
                    elif right_gesture == "two" and prev_right_gesture != "two": #minimize active window
                        # pyautogui.hotkey('command', 'm')
                        if system_type == "Mac":
                            pyautogui.hotkey('command', 'm')
                        elif system_type == "Windows":
                            pyautogui.hotkey('super', 'm')
                    elif right_gesture == "three" and prev_right_gesture != "three": #decrease text size
                        # pyautogui.hotkey('command', 'm')
                        if system_type == "Mac":
                            pyautogui.hotkey('command', '-')
                        elif system_type == "Windows":
                            pyautogui.hotkey('ctrl', '-')
                    elif right_gesture == "four" and prev_right_gesture != "four": #increase text size
                         # pyautogui.hotkey('command', 'm')
                        if system_type == "Mac":
                            pyautogui.hotkey('command', '+')
                        elif system_type == "Windows":
                            pyautogui.hotkey('ctrl', '+')
                #changed safari mode to browser mode
                elif mode == 'browser':
                    # if time_start is None:
                    #     time_start = time.time()
                    # time_stamp = time.time()
                    # if (time_stamp - time_start > 1):
                    #     time_start = time_stamp
                    if right_gesture == "one" and prev_right_gesture != "one": # new tab
                        if system_type == "Mac":
                            keyboard.press(Key.cmd)
                            keyboard.press('t')
                            keyboard.release('t')
                            keyboard.release(Key.cmd)
                        elif system_type == "Windows":
                            keyboard.press(Key.ctrl)
                            keyboard.press('t')
                            keyboard.release('t')
                            keyboard.release(Key.ctrl)
                        #time.sleep(0.5)
                        
                    if right_gesture == "two" and prev_right_gesture != "two": # address bar
                        if system_type == "Mac":
                            keyboard.press(Key.cmd)
                            keyboard.press('l')
                            keyboard.release('l')
                            keyboard.release(Key.cmd)
                        elif system_type == "Windows":
                            keyboard.press(Key.ctrl)
                            keyboard.press('l')
                            keyboard.release('l')
                            keyboard.release(Key.ctrl)
                        #time.sleep(0.5)
                    if right_gesture == "three" and prev_right_gesture != "three": #close tab
                        if system_type == "Mac":
                            keyboard.press(Key.cmd)
                            keyboard.press('w')
                            keyboard.release('w')
                            keyboard.release(Key.cmd)
                        elif system_type == "Windows":
                            keyboard.press(Key.ctrl)
                            keyboard.press('w')
                            keyboard.release('w')
                            keyboard.release(Key.ctrl)
                        #time.sleep(0.5)
                        # if right_gesture == "four": # decrease text size
                        #     if system_type == "Mac":
                        #         keyboard.press(Key.cmd)
                        #         keyboard.press('-')
                        #         keyboard.release('-')
                        #         keyboard.release(Key.cmd)
                        #     elif system_type == "Windows":
                        #         keyboard.press(Key.ctrl)
                        #         keyboard.press('-')
                        #         keyboard.release('-')
                        #         keyboard.release(Key.ctrl)
                        #     #time.sleep(0.5)
                        
                        # if right_gesture == "five": # increase text size
                        #     if system_type == "Mac":
                        #         keyboard.press(Key.cmd)
                        #         keyboard.press('+')
                        #         keyboard.release('+')
                        #         keyboard.release(Key.cmd)
                        #     elif system_type == "Windows":
                        #         keyboard.press(Key.ctrl)
                        #         keyboard.press('+')
                        #         keyboard.release('+')
                        #         keyboard.release(Key.ctrl)
                        #     #time.sleep(0.5)
                            
                    if right_gesture == 'arrow' and prev_right_gesture != "arrow": # switch tab
                        if system_type == "Mac":
                            keyboard.press(Key.ctrl)
                            keyboard.press(Key.tab)
                            keyboard.release(Key.tab)
                            keyboard.release(Key.ctrl)
                        elif system_type == "Windows":
                            keyboard.press(Key.ctrl)
                            keyboard.press(Key.tab)
                            keyboard.release(Key.tab)
                            keyboard.release(Key.ctrl)
                        #time.sleep(0.5)
            
# ------------------ END MOUSE MODE ------------------
# ------------------ BEGIN KEYBOARD MODE ------------------
        else:
            # Initialize the Drawn Image into a new image and display window
            if drawn_image is None:
                drawn_image = np.zeros(image.shape[0:2], dtype=np.uint8)
                # cv2.imshow('Drawn Image', drawn_image)

            # if right hand is pose 'one', append the fingertip
            # if right hand is not pose 'one', pause the fingertip trace
            if right_gesture != 'one':
                new_fingertip_segment = []
                new_fingertip_segment_appended = False
            if right_gesture == 'one':
                x_pixel = int(right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x\
                                * image_width)

                y_pixel = int(right_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y\
                                * image_height)
                new_fingertip_segment.append((x_pixel, y_pixel))
                if new_fingertip_segment_appended and len(fingertip_path_right) > 0:
                    fingertip_path_right[-1] = new_fingertip_segment
                else:
                    fingertip_path_right.append(new_fingertip_segment)
                    new_fingertip_segment_appended = True

            # if right hand is a fist, show path on rising edge and reset path
            if right_gesture == 'fist' and prev_right_gesture != 'fist':
                # on rising edge, draw path in separate window
                drawn_path_resized = keyboard_util.crop_and_draw_path(drawn_image, fingertip_path_right)

                if drawn_path_resized is not None:
                    # get the correct model with respect to the input_mode
                    if input_mode == 1:
                        active_model = char_model
                        active_map = char_map
                        tensor_input_image_reshaped = drawn_path_resized.T.reshape((1, 1, 28, 28))
                    else:
                        active_model = digit_model
                        active_map = digit_map
                        tensor_input_image_reshaped = drawn_path_resized.reshape((-1, 28*28))

                    tensor_input_image = torch.from_numpy(tensor_input_image_reshaped) / 255
                    character_confidences = active_model(tensor_input_image.to(DEVICE))

                    # normalize output for probabilities if in number mode
                    if input_mode == -1:
                        character_confidences = torch.nn.functional.softmax(character_confidences)
                    
                    top2 = torch.topk(character_confidences, 2).indices.tolist()[0]
                    probs = torch.topk(character_confidences, 2).values.tolist()[0]

                    #If the top two probabilities are above a threshold, enter confidence selection stage
                    if probs[1] >= 0.05: #thresholds can be tuned.
                        choice1 = active_map[int(top2[0])]
                        choice2 = active_map[int(top2[1])]
                        print("CONFIDENCE SELECTION: " + "\n")
                        print(f'Top Choice 1: {active_map[int(top2[0])]} with probability {probs[0]}')
                        print(f'Top Choice 2: {active_map[int(top2[1])]} with probability {probs[1]}')
                    else:
                        _, label = torch.max(character_confidences, axis=1)
                        print(f'Recognized character: {active_map[int(label)]}')
                        keyboard.press(active_map[int(label)])
                        keyboard.release(active_map[int(label)])

                    

                # reset path once submission has been processed
                fingertip_path_right = []
                new_fingertip_segment = []
                new_fingertip_segment_appended = False

            # if right hand is a four, discard drawing
            if right_gesture == "four" and prev_right_gesture != "four":
                fingertip_path_right = []
                new_fingertip_segment = []
                new_fingertip_segment_appended = False

            # get hand position (as a percentage)
            if left_landmarks is not None:
                hand_pos_percent = keyboard_util.modifiers_hand_position(mp_hands, left_landmarks)

                #if we are currently in the confidence selection stage, then check the user's hand position
                if choice1 != None and choice2 != None:

                    if choice1_start_percentage is None:
                        image, choice1_start_percentage, \
                        choice1_end_percentage, \
                        choice2_start_percentage, choice2_end_percentage = \
                            keyboard_util.confidence_selection(image, choice1, choice2)

                    if hand_pos_percent[0] > choice1_start_percentage[0] and \
                            hand_pos_percent[0] < choice1_end_percentage[0] and \
                            hand_pos_percent[1] > choice1_start_percentage[1] \
                            and hand_pos_percent[1] < choice1_end_percentage[1]:
                        choice_made = choice1
                    if hand_pos_percent[0] > choice2_start_percentage[0] \
                            and hand_pos_percent[0] < choice2_end_percentage[0] and \
                            hand_pos_percent[1] > choice2_start_percentage[1] and \
                            hand_pos_percent[1] < choice2_end_percentage[1]:
                        choice_made = choice2

                    #If they have made a choice, then press the key and exit confidence selection
                    if choice_made != None:
                        print(f'Chosen character: {choice_made}')
                        keyboard.press(choice_made)
                        keyboard.release(choice_made)
                        choice1 = None
                        choice2 = None
                        choice_made = None
                
                # if not in character selection stage, space/backspace available
                else:
                    # if in backspace box, backspace once
                    if hand_pos_percent[0] < bksp_space_end_percentage[0] and hand_pos_percent[1]\
                                                                        < bksp_space_end_percentage[1]:
                        if not backspace_available:
                            keyboard.press(Key.backspace)
                            keyboard.release(Key.backspace)
                            backspace_available = True
                
                    # once hand leaves the box, make backspace possible again
                    else:
                        backspace_available = False
                    
                    # if in space box, space once
                    if hand_pos_percent[0] < bksp_space_end_percentage[0] and hand_pos_percent[1]\
                                                                        > bksp_space_end_percentage[1]:
                        if not space_available:
                            keyboard.press(Key.space)
                            keyboard.release(Key.space)
                            space_available = True
                
                    # once hand leaves the box, make space possible again
                    else:
                        space_available = False

            # backspace once using gesture
            if left_gesture == "one_left" and prev_left_gesture != "one_left":
                keyboard.press(Key.backspace)
                keyboard.release(Key.backspace)

            if left_gesture == 'four_left' and prev_left_gesture != "four_left":
                input_mode *= -1

            # TODO: we might want to look into border box / some other indication instead of text
            if input_mode == 1:
                cv2.putText(image, 'input mode: char', (int(image_width * 0.25),
                                                        int(image_height * 0.9)),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                        (209, 80, 0, 255), 3)
            else:
                cv2.putText(image, 'input mode: digit', (int(image_width * 0.25),
                                                         int(image_height * 0.9)),
                                                         cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                         (209, 80, 0, 255), 3)

        # draw the path of the fingertip
        if len(fingertip_path_right) > 0:
            for points_arr in fingertip_path_right:
                if len(points_arr) > 0 :
                    image = keyboard_util.draw_path(image, points_arr)      

        if right_rising_edge_gesture:
            prev_right_gesture = right_gesture
            right_rising_edge_gesture = False

        if left_rising_edge_gesture:
            prev_left_gesture = left_gesture
            left_rising_edge_gesture = False

    else:
        fingertip_path_right = []
# ------------------ END KEYBOARD MODE ------------------

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow(win_name, image)
    # cv2.resizeWindow(win_name, 320, 180)

    # # calculate latency
    # tick_end = cv2.getTickCount()
    # latency = (tick_end - tick_start) / cv2.getTickFrequency()
    # print('Latency: {} seconds'.format(latency))
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
