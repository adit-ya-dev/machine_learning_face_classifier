import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # Load pre-trained face detection cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_face(self, image):
        """
        Detect faces in the input image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            faces: List of detected face regions (x, y, w, h)
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def extract_face_roi(self, image, face):
        """
        Extract face region of interest
        
        Args:
            image: Input image
            face: Face coordinates (x, y, w, h)
            
        Returns:
            face_roi: Extracted face region
        """
        x, y, w, h = face
        face_roi = image[y:y+h, x:x+w]
        return face_roi
