import numpy as np
import tensorflow as tf
import cv2
import time
from pushbullet import Pushbullet
from imutils.video import VideoStream
from gpiozero import Buzzer
from time import sleep


drowsy_frame_count = 0
drowsy_threshold = 10

buzzer = Buzzer(26)

class_names = open("labels.txt", "r").readlines()

api_key = 'o.8A3yoAGz6eiHfiySQihP4rQPNqNTEY03'
pb = Pushbullet(api_key)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()


class_names = [str(i) for i in range(interpreter.get_output_details()[0]['shape'][-1])]


vs = VideoStream(src=1,framerate=10).start()
time.sleep(2.0)

#notification initialization
def send_notification(title, message):
    push = pb.push_note(title, message)


while True:
    # Capture a frame from the USB webcam
    frame = vs.read()
  
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Reshape the image for the TFLite model input
    resized_frame = resized_frame.reshape(1, 224, 224, 3).astype(np.float32)
    # Normalize the pixel values
    normalized_frame = resized_frame / 255.0

    normalized_frame = normalized_frame.astype(np.float32)

    # Set the input tensor
    input_tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_tensor_index, normalized_frame)

    # Run inference
    interpreter.invoke()

    # Get the model output
    output_tensor_index = interpreter.get_output_details()[0]['index']
    predictions = interpreter.get_tensor(output_tensor_index)[0]

    # Get the predicted class and confidence score
    predicted_class = np.argmax(predictions)
    print(predictions)
    confidence_score = predictions[predicted_class]


    if predicted_class == 1:
        drowsy_frame_count += 1
        print("Drowsy Frame Count:", drowsy_frame_count)

        if drowsy_frame_count >= drowsy_threshold:
            title = 'High Sleep Score Alert'
            message = 'Driver seems asleep. Take necessary action!'
            send_notification(title, message)
            print('get some sleep')

            buzzer.on()
            sleep(1)
            buzzer.off()
            sleep(1)
    else:
          drowsy_frame_count = 0



