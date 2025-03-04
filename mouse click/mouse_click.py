import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Variables for smoothing
smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    img = cv2.flip(img, 1)

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(img_rgb)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmarks for the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            # Get the landmarks for the thumb tip (landmark 4)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert the coordinates to screen dimensions
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Smooth the coordinates to reduce jitter
            curr_x = prev_x + (x - prev_x) / smoothening
            curr_y = prev_y + (y - prev_y) / smoothening

            # Move the mouse to the smoothed coordinates
            pyautogui.moveTo(curr_x, curr_y)

            # Update previous coordinates
            prev_x, prev_y = curr_x, curr_y

            # Calculate the distance between the index finger tip and thumb tip
            distance = ((index_finger_tip.x - thumb_tip.x)**2 + (index_finger_tip.y - thumb_tip.y)**2)**0.5

            # If the distance is small, perform a mouse click
            if distance < 0.05:
                pyautogui.click()
                time.sleep(0.2)  # Add a small delay to avoid multiple clicks

    # Display the image
    cv2.imshow("AI Virtual Mouse", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()