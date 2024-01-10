import pathlib
import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('keras_model.h5')

# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_files = pathlib.Path('pretrainedmodel.tflite')
tflite_model_files.write_bytes(tflite_model)