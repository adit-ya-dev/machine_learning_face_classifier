# Sample Images for Age Classification

This directory is for storing test images for the age classification system.

## Image Requirements:
- Format: JPG or PNG
- Contains at least one clearly visible face
- Good lighting conditions
- Frontal face view preferred
- Recommended size: At least 300x300 pixels

## Usage:
1. Place your test images in this directory
2. Run the classifier with:
   ```
   python main.py samples/your_image.jpg
   ```

## Testing Process:
1. The system will detect faces in the image
2. For each detected face, it will:
   - Extract facial features
   - Classify the age group
   - Display results with confidence score
   - Show the image with detected faces marked
