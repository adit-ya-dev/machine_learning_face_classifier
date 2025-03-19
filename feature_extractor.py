cv2
import numpy as np
from scipy.stats import moment
import
class FeatureExtractor:
    def __init__(self):
        # Define feature extraction parameters
        self.target_size = (64, 64)
        
    def extract_features(self, face_roi):
        """
        Extract features from face ROI
        
        Args:
            face_roi: Face region of interest
            
        Returns:
            features: Extracted feature vector
        """
        # Resize face ROI to standard size
        face_resized = cv2.resize(face_roi, self.target_size)
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features
        hog_features = self._extract_hog(gray)
        
        # Extract LBP features
        lbp_features = self._extract_lbp(gray)
        
        # Extract statistical features
        stat_features = self._extract_statistical_features(gray)
        
        # Combine all features
        features = np.concatenate([hog_features, lbp_features, stat_features])
        
        return features
    
    def _extract_hog(self, gray):
        """Extract HOG features"""
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(gray)
        return hog_features.flatten()
    
    def _extract_lbp(self, gray):
        """Extract LBP features"""
        radius = 1
        n_points = 8
        
        lbp = np.zeros_like(gray)
        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                pattern = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = i + radius * np.cos(angle)
                    y = j - radius * np.sin(angle)
                    pattern |= (gray[int(x), int(y)] > center) << k
                lbp[i, j] = pattern
                
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return hist
    
    def _extract_statistical_features(self, gray):
        """Extract statistical features"""
        mean = np.mean(gray)
        std = np.std(gray)
        skewness = moment(gray.ravel(), moment=3)
        kurtosis = moment(gray.ravel(), moment=4)
        
        return np.array([mean, std, skewness, kurtosis])
