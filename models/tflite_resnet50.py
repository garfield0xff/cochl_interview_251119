import tensorflow as tf

model = tf.keras.applications.ResNet50(weights="imagenet")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("resnet50.tflite", "wb") as f:
    f.write(tflite_model)
