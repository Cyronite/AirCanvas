import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
video = cv2.VideoCapture(0)
Lines = []
erasing = False
Squares = []

tools = {
    "Red Marker": (0, 0, 255),
    "Green Marker": (0, 255, 0),
    "Blue Marker": (255, 0, 0),
    "Eraser": (0, 0, 0),
    "Blue Square": (255, 0, 0)
}
selected_tool = "Blue Marker"

cv2.namedWindow('video', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = video.read()

    frame = flipped_frame_horizontal = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    h, w, c = frame.shape
    
    RedMarker = cv2.imread("Assets/RedMarker.png")
    GreenMarker = cv2.imread("Assets/GreenMarker.png")
    BlueMarker = cv2.imread("Assets/BlueMarker.png")
    Eraser = cv2.imread("Assets/Eraser.png")
    BlackScreen = cv2.imread("Assets/BlackScreen.png")
    SquareShape = cv2.imread("Assets/SquareShape.png")
    BlackScreen = cv2.resize(BlackScreen, (960, 85))
    RedMarker = cv2.resize(RedMarker, (85,85))
    GreenMarker = cv2.resize(GreenMarker, (85,85))
    BlueMarker = cv2.resize(BlueMarker, (85,85))
    Eraser = cv2.resize(Eraser, (85,85))
    SquareShape = cv2.resize(SquareShape, (85, 85)) 
    frame[0:85, 0:960] = BlackScreen
    frame[0:85, 500:585] = RedMarker
    frame[0:85, 600:685] = GreenMarker
    frame[0:85, 700:785] = BlueMarker
    frame[0:85, 800:885] = Eraser
    frame[0:85, 400:485] = SquareShape

    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
           
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            h, w, _ = frame.shape
            thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
        
            distance =  np.linalg.norm(np.array(thumb_tip_coords) - np.array(index_finger_tip_coords))
            touch_threshold = 35

            middle = (int(np.round((thumb_tip_coords[0] + index_finger_tip_coords[0])/2 , decimals=0)), int(np.round((thumb_tip_coords[1] + index_finger_tip_coords[1])/2 , decimals=0)) ) 

            if erasing:
                selected_tool = 'Eraser'
            elif selected_tool in tools:
                selected_tool = selected_tool
            else:
                selected_tool = 'Unknown'

            cv2.putText(frame, f'Selected: {selected_tool}', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
            if distance < touch_threshold:
                cv2.putText(frame, 'Touching', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if erasing == False:
                    if 'Square' in selected_tool:
                        if 'start_point' not in locals():
                            start_point = middle
                        end_point = middle
                    else:
                        if len(Lines) == 0 or np.linalg.norm(np.array(Lines[-1][-1]) - np.array(middle)) >= 30:
                            Lines.append([tools[selected_tool], middle])
                        else:
                            Lines[-1].append(middle)
                else:
                    cv2.circle(frame, middle, 30, (0, 0, 0), -1)
                    for line in Lines[:]:
                        cur_color = line[0]
                        if any((middle[0] - 30 <= point[0] <= middle[0] + 30 and middle[1] - 30 <= point[1] <= middle[1] + 30) for point in line[1:]):
                            Lines.remove(line)
                            
            else:
                cv2.putText(frame, 'Not Touching', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if 0 <= index_finger_tip_coords[1] <= 85:
                    if 500 <= index_finger_tip_coords[0] <= 585:
                        selected_tool = "Red Marker"
                        erasing = False
                    elif 600 <= index_finger_tip_coords[0] <= 685:
                        selected_tool = "Green Marker"
                        erasing = False
                    elif 700 <= index_finger_tip_coords[0] <= 785:
                        selected_tool = "Blue Marker"
                        erasing = False
                    elif 800 <= index_finger_tip_coords[0] <= 885:
                        erasing = True
                    elif 400 <= index_finger_tip_coords[0] <= 485:
                        selected_tool = "Red Square"
                        erasing = False
                if 'start_point' in locals():
                    Squares.append((start_point, end_point, tools[selected_tool]))
                    del start_point
    else:
        cv2.putText(frame, 'No Hand Detected', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    for line in Lines:
        if len(line) > 2:
            cur_color = line[0]
            line.pop(0)
            newLine = np.array(line)
            cv2.polylines(frame, [newLine], isClosed=False, color=cur_color, thickness=5)
            line.insert(0, cur_color)

    for square in Squares:
        cv2.rectangle(frame, square[0], square[1], square[2], -1)

    if 'start_point' in locals() and 'Square' in selected_tool:
        cv2.rectangle(frame, start_point, end_point, tools[selected_tool], -1)

    if len(Lines) > 0 and not erasing and 'Square' not in selected_tool:
        cur_color = Lines[-1][0]
        newLine = np.array(Lines[-1][1:])
        cv2.polylines(frame, [newLine], isClosed=False, color=cur_color, thickness=5)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) != -1:
        break
   
video.release()
cv2.destroyAllWindows()