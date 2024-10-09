import cv2
import matplotlib.pyplot as plt

# Load the pre-trained Haar Cascade model for face detection (You can replace with other cascades)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load an image
img_path = img_path = 'E:/sai/UCE/Video_summarization/object detection/human.jpg'   #Replace with the path to your image
img = cv2.imread(filename='E:/sai/UCE/Video_summarization/object detection/human.jpg')

# Convert to grayscale (Haar Cascades work better on grayscale images)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform object detection
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw bounding boxes around detected objects (faces in this case)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axis
plt.show()

# Optionally save the output image
cv2.imwrite('detected_faces.jpg', img)
