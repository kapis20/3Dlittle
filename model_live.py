import tflite_runtime.interpreter as tflite
import numpy as np


import cv2

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Preprocess the frame
    input_shape = input_details[0]['shape']  # Get model's input shape
    input_height, input_width = input_shape[1], input_shape[2]

    # Resize and normalize the frame
    input_frame = cv2.resize(frame, (input_width, input_height))
    input_frame = input_frame.astype(np.float32) / 255.0  # Normalize if needed
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension

    # Set the tensor for the input
    interpreter.set_tensor(input_details[0]['index'], input_frame)

    # Run inference
    interpreter.invoke()

    # Get predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Process predictions (example: assuming classification)
    predicted_class = np.argmax(predictions)
    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Prediction", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
