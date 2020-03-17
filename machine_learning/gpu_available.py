import time
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)


ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

x = tf.random_uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))


def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)

    result = time.time()-start

    print("10 loops: {:0.2f}ms".format(1000*result))


# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
    # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    with tf.device("GPU:0"):
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)
