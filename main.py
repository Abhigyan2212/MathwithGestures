import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import time
import streamlit as st

# Streamlit setup
st.set_page_config(layout='wide')
st.image('MathGestures.png')

col1, col2 = st.columns([2, 1])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.empty()

# Google generative AI setup
genai.configure(api_key="AIzaSyDyEwDcAvAYkA9HhE2aSEUkjB-PTpcYcwI")  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None, None

def draw(info, prev_pos, canvas, img):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # Only draw when the index finger is up
        current_pos = lmList[8][0:2]  # Get the coordinates of the index fingertip
        if prev_pos is not None:  # Only draw if there is a previous position
            cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
        prev_pos = current_pos  # Update the previous position
    elif fingers == [1, 1, 1, 1, 1]:  # Reset canvas if all fingers are up
        canvas = np.zeros_like(img)

    return prev_pos, canvas

def sendtoAI(model, canvas):
    pil_image = Image.fromarray(canvas)
    response = model.generate_content(["Solve this Math Problem", pil_image])
    return response.text

prev_pos = None
canvas = None
drawing = False
stop_drawing_time = None
drawing_stopped_threshold = 2  # seconds

# Continuously get frames from the webcam
while run:
    # Capture each frame from the webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if success:
        if canvas is None:
            canvas = np.zeros_like(img)  # Initialize the canvas once

        info = getHandInfo(img)

        if info[0] is not None:
            fingers, lmList = info
            if fingers == [0, 1, 0, 0, 0]:
                prev_pos, canvas = draw(info, prev_pos, canvas, img)
                drawing = True
                stop_drawing_time = None  # Reset stop drawing time
            elif fingers == [1, 1, 1, 1, 1]:
                canvas = np.zeros_like(img)  # Reset the canvas when all fingers are up
                prev_pos = None  # Reset previous position to stop line drawing
            else:
                prev_pos = None  # Reset previous position to stop line drawing
                if drawing:
                    if stop_drawing_time is None:
                        stop_drawing_time = time.time()
                    elif time.time() - stop_drawing_time > drawing_stopped_threshold:
                        response_text = sendtoAI(model, canvas)
                        if response_text:
                            output_text_area.subheader(response_text)  # Update the AI response in Streamlit
                            run = False  # Stop the loop once the drawing stops

        # Combine the canvas and the image
        image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(image_combined, channels="BGR")

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
