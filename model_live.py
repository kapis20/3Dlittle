# import cv2
# import numpy as np
# import tensorflow as tf

# # Path to the TFLite model
# model_path = 'model.tflite'

# # Single class name
# class_name = "spaghetti"

# # Define a single color for visualization
# COLOR = (0, 255, 0)  # Green

# def preprocess_frame(frame, input_size):
#     """Preprocess the input frame to feed to the TFLite model."""
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#     img = tf.image.resize(img, input_size)
#     img = img[tf.newaxis, :]  # Add batch dimension
#     img = tf.cast(img, dtype=tf.uint8)  # Ensure uint8 type
#     return img


# def detect_objects(interpreter, image, threshold):
#     """Run object detection and return the results."""
#     signature_fn = interpreter.get_signature_runner()
#     output = signature_fn(images=image)

#     # Extract detection results
#     count = int(np.squeeze(output['output_0']))
#     scores = np.squeeze(output['output_1'])
#     boxes = np.squeeze(output['output_3'])

#     results = []
#     for i in range(count):
#         if scores[i] >= threshold:
#             result = {
#                 'bounding_box': boxes[i],
#                 'score': scores[i]
#             }
#             results.append(result)
#     return results


# def run_detection_on_frame(frame, interpreter, threshold=0.5):
#     """Run object detection and annotate the frame with results."""
#     # Get input details
#     _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

#     # Preprocess the frame
#     preprocessed_frame = preprocess_frame(frame, (input_height, input_width))

#     # Run object detection
#     results = detect_objects(interpreter, preprocessed_frame, threshold)

#     # Annotate the frame
#     for obj in results:
#         # Convert bounding box to absolute coordinates
#         ymin, xmin, ymax, xmax = obj['bounding_box']
#         xmin = int(xmin * frame.shape[1])
#         xmax = int(xmax * frame.shape[1])
#         ymin = int(ymin * frame.shape[0])
#         ymax = int(ymax * frame.shape[0])

#         # Draw bounding box and label
#         label = f"{class_name}: {obj['score']:.2f}"
#         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), COLOR, 2)
#         cv2.putText(frame, label, (xmin, ymin - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
#     return frame


# def main():
#     # Load the TFLite model
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()

#     # Start video capture
#     cap = cv2.VideoCapture(0)  # Use 0 for the default camera

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Failed to capture frame.")
#                 break

#             # Run object detection on the frame
#             annotated_frame = run_detection_on_frame(frame, interpreter, threshold=0.5)

#             # Display the annotated frame
#             cv2.imshow("Object Detection - Spaghetti", annotated_frame)

#             # Exit on pressing 'q'
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         # Release resources
#         cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
import cv2
print(cv2.__version__)
