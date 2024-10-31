import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Define the paths for classified images
base_path = "F:\\coco"
large_path = os.path.join(base_path, "A1")
medium_path = os.path.join(base_path, "B1")
small_path = os.path.join(base_path, "C1")

# Create directories if they don't exist
os.makedirs(large_path, exist_ok=True)
os.makedirs(medium_path, exist_ok=True)
os.makedirs(small_path, exist_ok=True)

# Function to capture an image from the camera
def capture_image():
    cap = cv2.VideoCapture(0)  # Initialize the camera
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        raise Exception("Failed to capture image")

# Sample data (sizes and labels)
# These should be replaced with real data collected from previous images
sample_sizes = np.array([9500, 10000, 12000, 6000, 6500, 7000, 3000, 3500, 4000])
labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # 0: large, 1: medium, 2: small

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(sample_sizes.reshape(-1, 1), labels)

def extract_size_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 50)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizes = []
    for contour in contours:
        if cv2.contourArea(contour) > 150:  # Filter out small areas
            # Compute the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            size = w * h
            sizes.append((x, y, w, h, size))
    return sizes

def save_image(image, x, y, w, h, size_class):
    timestamp = int(time.time() * 1000)  # Get current timestamp in milliseconds
    if size_class == 0:
        folder = large_path
    elif size_class == 1:
        folder = medium_path
    else:
        folder = small_path

    filename = os.path.join(folder, f"coconut_{timestamp}.jpg")
    cv2.imwrite(filename, image)
    print(f"Saved to {filename}")

def evaluate_knn_on_directory(path, true_label):
    predicted_labels = []
    true_labels = []

    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            sizes = extract_size_from_image(image)

            for _, _, _, _, size in sizes:
                predicted_label = knn.predict([[size]])[0]
                predicted_labels.append(predicted_label)
                true_labels.append(true_label)

    if len(true_labels) == 0:
        return 0  # Return 0 accuracy if no images are processed

    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

def plot_accuracy(accuracies, categories):
    plt.figure(figsize=(10, 6))
    plt.bar(categories, accuracies, color=['blue', 'green', 'red'])
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of KNN Classifier for Coconut Sizes')
    plt.ylim(0, 1)
    plt.show()

def main():
    true_labels = []
    predicted_labels = []

    while True:
        image = capture_image()
        height, width = image.shape[:2]
        # Define the region of interest (ROI) in the center of the image
        roi_x1, roi_y1 = width // 4, height // 4
        roi_x2, roi_y2 = width * 3 // 4, height * 3 // 4

        # Draw the grid and ROI rectangle
        cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        grid_color = (255, 255, 255)
        for i in range(1, 4):
            cv2.line(image, (i * width // 4, 0), (i * width // 4, height), grid_color, 1)
            cv2.line(image, (0, i * height // 4), (width, i * height // 4), grid_color, 1)

        # Extract sizes from the center region only
        center_image = image[roi_y1:roi_y2, roi_x1:roi_x2]
        sizes = extract_size_from_image(center_image)

        for x, y, w, h, size in sizes:
            size_class = knn.predict([[size]])[0]
            save_image(image, x, y, w, h, size_class)
            true_labels.append(size_class)
            predicted_labels.append(size_class)

        # Display the captured image with grid and center rectangle
        cv2.imshow("Captured Image", image)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Calculate and print the accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall Accuracy: {accuracy}")

    # Calculate and plot accuracies for each category
    accuracy_large = evaluate_knn_on_directory(large_path, 0)
    accuracy_medium = evaluate_knn_on_directory(medium_path, 1)
    accuracy_small = evaluate_knn_on_directory(small_path, 2)

    # Plot accuracies
    accuracies = [accuracy_large, accuracy_medium, accuracy_small]
    categories = ['Large (A1)', 'Medium (B1)', 'Small (C1)']

    plot_accuracy(accuracies, categories)

    print(f"Accuracy for large coconuts (A1): {accuracy_large}")
    print(f"Accuracy for medium coconuts (B1): {accuracy_medium}")
    print(f"Accuracy for small coconuts (C1): {accuracy_small}")

if __name__ == "__main__":
    main()
