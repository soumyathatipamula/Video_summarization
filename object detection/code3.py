import cv2
import matplotlib.pyplot as plt

# Load the pre-trained Haar Cascade model for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load an image
img_path = r'E:\sai\UCE\Video_summarization\object detection\human.jpg'  # Replace with the correct path to your image
img = cv2.imread(img_path)

img_path2 = r'E:\sai\UCE\Video_summarization\object detection\dog.jpg'
img = cv2.imread(img_path2)

# Check if the image was loaded correctly
if img is None:
    print(f"Error: Unable to load image at {img_path}")
else:
    # Convert to grayscale (Haar Cascades work better on grayscale images)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes and label 'Human' for each detected face
    for (x, y, w, h) in faces:
        # Draw the bounding box around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Add the 'Human' label above the bounding box
        cv2.putText(img, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image with bounding boxes and labels
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis
    plt.show()

    # Optionally save the output image with detected faces and labels
    cv2.imwrite('detected_faces_with_labels.jpg', img)
