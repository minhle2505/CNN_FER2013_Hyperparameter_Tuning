import tensorflow as tf

# Check the TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# List available physical devices
physical_devices = tf.config.list_physical_devices()
print(f"Physical devices: {physical_devices}")

# Check if GPU is available
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU devices: {gpu_devices}")