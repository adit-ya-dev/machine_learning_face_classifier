import cv2
import numpy as np

def load_image(path):
    """
    Load image from path
    
    Args:
        path: Image file path
        
    Returns:
        image: Loaded image
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image from {path}")
    return image

def preprocess_image(image):
    """
    Preprocess image for face detection
    
    Args:
        image: Input image
        
    Returns:
        processed_image: Preprocessed image
    """
    # Resize if image is too large
    max_size = 1024
    height, width = image.shape[:2]
    
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        image = cv2.resize(image, new_size)
    
    # Normalize pixel values
    image = image.astype(np.float32) / 255.0
    
    return image

def age_to_group(age):
    """
    Convert numerical age to age group
    
    Args:
        age: Numerical age
        
    Returns:
        group: Age group label
    """
    if age < 18:
        return 0  # Child
    elif age < 30:
        return 1  # Young Adult
    elif age < 45:
        return 2  # Adult
    elif age < 60:
        return 3  # Middle Aged
    else:
        return 4  # Senior
