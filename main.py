        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        print("Please ensure your image is in the 'samples' directory.")
        return

    try:
        # Load and process image
        print(f"\nProcessing image: {image_path}")
        image = load_image(image_path)
        processed_image = preprocess_image(image)

        # Detect face
        faces = face_detector.detect_face(processed_image)

        if len(faces) == 0:
            print("No faces detected in the image.")
            return

        print(f"Detected {len(faces)} face(s)")

        # Process first detected face
        face_roi = face_detector.extract_face_roi(processed_image, faces[0])
        features = feature_extractor.extract_features(face_roi)

        # Make prediction
        features_reshaped = features.reshape(1, -1)
        predicted_age_group = classifier.predict(features_reshaped)[0]
        confidence = classifier.get_confidence(features_reshaped)[0]

        # Display results
        age_group_names = {
            0: "Child (<18)",
            1: "Young Adult (18-29)",
            2: "Adult (30-44)",
            3: "Middle Aged (45-59)",
            4: "Senior (60+)"
        }

        print("\nResults:")
        print(f"Predicted Age Group: {age_group_names[predicted_age_group]}")
        print(f"Confidence: {confidence:.2f}")

        # Draw rectangle around detected face
        x, y, w, h = faces[0]
        result_image = processed_image.copy()
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Add text with prediction
        text = f"{age_group_names[predicted_age_group]}"
        cv2.putText(result_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Age Classification Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
