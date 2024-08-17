import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# count fingers
def count_fingers(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    thumb_open = thumb_tip < thumb_ip if hand_landmarks.landmark[0].x < thumb_tip else thumb_tip > thumb_ip

    fingers_open = [thumb_open]
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]
    
    for tip in finger_tips:
        tip_y = hand_landmarks.landmark[tip].y
        dip_y = hand_landmarks.landmark[tip - 2].y
        fingers_open.append(tip_y < dip_y)

    return sum(fingers_open)

# Initialize camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

# tkinter to show image
root = tk.Tk()
root.title("Mister Programmer")
root.geometry("800x600")

# Create a black image for covering image
black_image = Image.new('RGBA', (800, 600), color=(0, 0, 0, 255))

# Load the show image and convert to RGBA for transparency control
show_image = Image.open("show.png").convert("RGBA")
show_image = show_image.resize((800, 600))

# Label to display images
label = tk.Label(root)
label.pack()

# Update function for Tkinter loop
def update_frame():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        root.after(10, update_frame)
        return
    
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Initialize the image to be shown
    output_image = black_image.copy()

    # If hands are detected in the frame
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            finger_count = count_fingers(hand_landmarks)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Fingers: {finger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Calculate opacity (0 to 1) based on the number of fingers count
            opacity = finger_count / 5.0

            if opacity > 0:
                output_image = Image.blend(black_image, show_image, opacity)

            output_image_tk = ImageTk.PhotoImage(output_image)
            label.config(image=output_image_tk)
            label.image = output_image_tk

    else:
        label.config(image=ImageTk.PhotoImage(black_image))

    # Show the frame in the OpenCV window
    cv2.imshow('Finger Counter', frame)

    root.after(10, update_frame)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    root.after(0, update_frame)
    root.mainloop()

cap.release()
cv2.destroyAllWindows()
