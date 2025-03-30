import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Load trained model
model = load_model("asl_fingerspelling_model.h5")  # Update with your model file name

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define ASL class labels (ensure this matches the labels used during training)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del"]

# Variables for phrase building
current_letter = None
letter_start_time = None
phrase = ""
letter_duration_threshold = 3  # Time in seconds to add a letter to the phrase

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)
            
            # Convert to NumPy array and reshape for model input
            landmarks = np.array(landmarks).reshape(1, -1)
            
            # Predict ASL letter
            prediction = model.predict(landmarks)
            predicted_index = np.argmax(prediction)
            predicted_letter = labels[predicted_index]
            
            # Check if the predicted letter is the same as the current letter
            if predicted_letter == current_letter:
            # Check if the letter has been shown for 5 seconds
                if time.time() - letter_start_time >= letter_duration_threshold:
                    # Handle special cases for "space" and "del"
                    if predicted_letter == "space":
                        phrase += " "  # Add a space to the phrase
                    elif predicted_letter == "del":
                        phrase = phrase[:-1]  # Remove the last character from the phrase
                    else:
                        phrase += predicted_letter  # Add the predicted letter to the phrase
                    
                    print(f"Phrase so far: {phrase}")
                    # Reset the current letter and start time
                    current_letter = None
                    letter_start_time = None
            else:
                # Update the current letter and start time
                current_letter = predicted_letter
                letter_start_time = time.time()
            
            # Display prediction on screen
            cv2.putText(frame, f"Predicted: {predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Phrase: {phrase}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Show output
    cv2.imshow("ASL Recognition", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()