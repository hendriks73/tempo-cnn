"""
tempo-cnn is a simple package that allows estimation of musical tempo.
"""
import logging
import tensorflow as tf

# reduce TensorFlow chatter
tf.get_logger().setLevel(logging.ERROR)
