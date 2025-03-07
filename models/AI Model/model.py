import cv2
import numpy as np
import pandas as pd

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()] 

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()] 

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize a dictionary to count occurrences of detected objects
object_count = {}
# Set to track detected labels in the current frame
detected_labels = set()

# Number of frames to consider for counting
frame_threshold = 5
frame_counter = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Process the outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Clear detected labels for the current frame
    current_frame_labels = set()

    # Draw bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Add label to current frame labels
            current_frame_labels.add(label)

    # Update counts based on current frame labels
    for label in current_frame_labels:
        if label not in detected_labels:
            detected_labels.add(label)
            if label in object_count:
                object_count[label] += 1
            else:
                object_count[label] = 1

    # Display the output frame
    cv2.imshow('Object Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Save the object counts to a CSV file using pandas
df = pd.DataFrame(list(object_count.items()), columns=['Object', 'Count'])
df.to_csv('detected_objects.csv', index=False)

print("Detected objects and their counts have been saved to 'detected_objects.csv'.")