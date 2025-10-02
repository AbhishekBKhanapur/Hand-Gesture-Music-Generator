import streamlit as st
import cv2
import mediapipe as mp
import simpleaudio as sa
import numpy as np
import math

# Streamlit page config
st.set_page_config(page_title="Hand Gesture Music Composer", layout="wide")

# Apply background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("bg.png");
background-size: cover;
background-position: center;
}
h1 {
color: cyan;
text-align: center;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.markdown("# HAND GESTURE BASED MUSIC COMPOSER")

# Start/Stop buttons
start = st.button("▶️ Start Camera")
stop = st.button("🛑 Stop Camera")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load sounds
sound1 = sa.WaveObject.from_wave_file("sound1.wav")
sound2 = sa.WaveObject.from_wave_file("sound2.wav")
sound3 = sa.WaveObject.from_wave_file("sound3.wav")

# Placeholder for camera frame
frame_display = st.image([])

# Camera logic
if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark
                index_thumb_distance = math.dist(
                    [landmarks[8].x, landmarks[8].y],
                    [landmarks[4].x, landmarks[4].y]
                )

                middle_thumb_distance = math.dist(
                    [landmarks[12].x, landmarks[12].y],
                    [landmarks[4].x, landmarks[4].y]
                )

                ring_thumb_distance = math.dist(
                    [landmarks[16].x, landmarks[16].y],
                    [landmarks[4].x, landmarks[4].y]
                )

                threshold = 0.05

                if index_thumb_distance < threshold:
                    sound1.play()
                elif middle_thumb_distance < threshold:
                    sound2.play()
                elif ring_thumb_distance < threshold:
                    sound3.play()

        frame_display.image(frame, channels="BGR")

    cap.release()
