import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import os
import math

# Streamlit app setup
st.title("AI Virtual Painter")
st.text("Interact with the AI Virtual Painter using hand gestures!")

# Mediapipe and OpenCV setup
mp_hands = mp.solutions.hands

# Configuration
width, height = 640, 480
draw_color = (0, 0, 255)  # Default color: Red
thickness = 15
tipIds = [4, 8, 12, 16, 20] # Fingertips indexes
xp, yp = 0, 0  # Previous points for drawing

# Load header images for color options
header_images = [f"Header/{img}" for img in os.listdir("Header") if img.endswith((".png", ".jpg", ".jpeg"))]
header_list = [cv2.imread(img) for img in header_images]
header = header_list[0]  # Default header

# Sidebar for color selection
# st.sidebar.title("Select Drawing Color")
color_map = {
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Eraser": (0, 0, 0)
}
selected_color = list(color_map.keys())[0]
draw_color = color_map[selected_color]

# Canvas for drawing
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Start capturing video
cap = cv2.VideoCapture(0)

# Create a placeholder for the live video feed
video_placeholder = st.empty()

# Process the video feed
with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = [
                    (int(lm.x * width), int(lm.y * height))
                    for lm in hand_landmarks.landmark
                ]

                if points:
                    x1, y1 = points[8]  # Index finger
                    x2, y2 = points[12]  # Middle finger
                    x3, y3 = points[4]  # Thumb
                    x4, y4 = points[20]  # Pinky finger

                     ## Checking which fingers are up
                    fingers = []
                    # Checking the thumb
                    if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # The rest of the fingers
                    for id in range(1, 5):
                        if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    # Check if in selection mode (two fingers up)
                    nonSel = [0, 3, 4] # indexes of the fingers that need to be down in the Selection Mode
                    if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in nonSel):
                        # Selection mode (change color)
                        xp, yp =[x1, y1] # Reset drawing position
                        
                        # Selecting the colors and the eraser on the screen
                        for i, img in enumerate(header_list):
                            if (10 < y1 < 100):
                                if (100 < x1 < 120):
                                    header = header_list[0]
                                    draw_color = color_map[list(color_map.keys())[0]]
                                elif (230 < x1 < 250):
                                    header = header_list[1]
                                    draw_color = color_map[list(color_map.keys())[1]]
                                elif (370 < x1 < 390):
                                    header = header_list[2]
                                    draw_color = color_map[list(color_map.keys())[2]]
                                elif (510 < x1 < 540):
                                    header = header_list[3]
                                    draw_color = color_map[list(color_map.keys())[3]]
                                break

                        cv2.rectangle(frame, (x1-10, y1-15), (x2+10, y2+23), draw_color, -1)

                    # Adjust line thickness using index finger and thumb
                    selecting = [1, 1, 0, 0, 0]  # Thumb and index finger up
                    setting = [1, 1, 0, 0, 1]    # Thumb, index, and pinky up
                    if all(fingers[i] == j for i, j in zip(range(0,5), selecting)) or \
                       all(fingers[i] == j for i, j in zip(range(0,5), setting)):

                        # Calculate radius based on distance between thumb and index finger
                        r = int(math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2) / 3)

                        # Find midpoint between thumb and index finger
                        x0, y0 = [(x1 + x3) / 2, (y1 + y3) / 2]

                        # Orthogonal vector for alignment
                        v1, v2 = [x1 - x3, y1 - y3]
                        v1, v2 = [-v2, v1]

                        # Normalize vector
                        mod_v = math.sqrt(v1 ** 2 + v2 ** 2)
                        v1, v2 = [v1 / mod_v, v2 / mod_v]

                        # Draw circle indicating thickness selection
                        c = 3 + r
                        x0, y0 = [int(x0 - v1 * c), int(y0 - v2 * c)]
                        cv2.circle(frame, (x0, y0), int(r / 2), draw_color, -1)

                        # Confirm thickness when pinky is up
                        if fingers[4]:
                            thickness = r
                            cv2.putText(frame, "Thickness Set", (x4 - 25, y4 - 8),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 1)

                        xp, yp = [x1, y1]

                    # Check if in drawing mode (only index finger up)
                    nonDraw = [0, 2, 3, 4]
                    if fingers[1] and all(fingers[i] == 0 for i in nonDraw):
                        cv2.circle(frame, (x1, y1), thickness // 2, draw_color, -1)
                        if xp == 0 and yp == 0:
                            xp, yp = [x1, y1]
                        cv2.line(canvas, (xp, yp), (x1, y1), draw_color, thickness)
                        xp, yp = [x1, y1]

                    # Check if erasing (hand closed - no fingers up)
                    if not any(fingers):
                        canvas = np.zeros((height, width, 3), dtype=np.uint8)
                        xp, yp = 0, 0

        # Overlay the canvas onto the video feed
        canvas_resized = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))

        # Create an inverted binary mask
        img_gray = cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

        # Combine the frame and the canvas
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, canvas_resized)

        # Add the header image to the frame
        frame_height, frame_width = frame.shape[:2]
        header_resized = cv2.resize(header, (frame_width, 100))
        frame[0:100, 0:frame_width] = header_resized

        # Update the live video feed in Streamlit
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

# Release the video capture
cap.release()
