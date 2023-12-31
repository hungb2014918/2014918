import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
model_path = 'D:/model_data/best_float32 (1).tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2
outname = output_details[0]['name']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for inference
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0)
    input_data = input_data.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("output data", output_data)

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
