import cv2
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        # Convert video frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for faster detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            try:
                # Predict emotion
                analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis[0]['dominant_emotion']

                # Draw bounding box and label
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, dominant_emotion.capitalize(), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception:
                pass

        return img

# Streamlit UI
st.title("Facial Emotion Recognition App")
st.write("Click **START** below to enable your webcam and analyze emotions in real time.")

webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetector)
