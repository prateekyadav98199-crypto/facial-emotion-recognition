import cv2
import streamlit as st
from fer import FER
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize detector outside loop for high performance
detector = FER(mtcnn=False)

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect emotions on frame
        results = detector.detect_emotions(img)
        
        for result in results:
            (x, y, w, h) = result["box"]
            emotions = result["emotions"]
            # Get the emotion with highest probability score
            dominant_emotion = max(emotions, key=emotions.get)
            
            # Draw bounding box and label
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, dominant_emotion.capitalize(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return img

st.title("Facial Emotion Recognition App")
st.write("Click **START** below to enable your webcam and analyze emotions in real time.")

webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetector)
