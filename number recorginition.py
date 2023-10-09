import cv2
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the digits dataset
digits = datasets.load_digits()

# Split the dataset into features and labels
X = digits.data
y = digits.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Initialize the camera (change the camera index as needed)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to 8x8 pixels
    small_gray = cv2.resize(gray, (8, 8))

    # Threshold and invert the colors
    _, small_gray = cv2.threshold(small_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Reshape the frame to match the input shape of the classifier
    input_image = small_gray.reshape(1, -1)

    # Predict the digit using the trained classifier
    predicted_digit = knn.predict(input_image)

    # Display the predicted digit on the frame
    cv2.putText(frame, str(predicted_digit[0]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Show the frame
    cv2.imshow('Digit Recognition', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
