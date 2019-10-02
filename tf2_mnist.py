from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tfds.load(name='mnist', split=['train', 'test'])
