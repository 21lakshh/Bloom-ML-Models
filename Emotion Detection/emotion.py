import cv2
from deepface import DeepFace
from collections import Counter
import time

def detect_emotion():
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start capturing video
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    # List to store detected emotions
    detected_emotions = []
    
    # Add frame counter
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
            
        frame_count += 1
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                try:
                    # Extract the face ROI (Region of Interest)
                    face_roi = rgb_frame[y:y + h, x:x + w]
                    
                    allowed_emotions = ['happy', 'angry', 'neutral']
                    # Analyze with DeepFace
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    # Get emotion predictions dictionary
                    emotion_predictions = result[0]['emotion']
                    # Filter out unwanted emotions
                    filtered_emotions = {k: v for k, v in emotion_predictions.items() if k in allowed_emotions}
                    # Determine the dominant emotion among the allowed ones
                    emotion = max(filtered_emotions, key=filtered_emotions.get)
                    
                    # Append the detected emotion to our list
                    detected_emotions.append(emotion)
                    
                    # Draw rectangle around face and label with predicted emotion
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    # Display the current emotion count
                    cv2.putText(frame, f'Detections: {len(detected_emotions)}', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display FPS
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display most common emotion so far
                    if detected_emotions:
                        emotion_counter = Counter(detected_emotions)
                        most_common = emotion_counter.most_common(1)[0]
                        cv2.putText(frame, f'Most common: {most_common[0]} ({most_common[1]})', 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Error in face analysis: {str(e)}")
                    continue
        else:
            # Display "No face detected" message
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)
            
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Return the most common emotion
    if detected_emotions:
        emotion_counter = Counter(detected_emotions)
        most_common_emotion = emotion_counter.most_common(1)[0][0]
        return most_common_emotion
    return None

if __name__ == "__main__":
    print("Starting emotion detection...")
    result = detect_emotion()
    print(f"Most common emotion detected: {result}")

