import tensorflow as tf

# TF 2.0 supports eager execution so don't explicitly create a session and run the code in it, instead you it the
# conventional way

# old boring/ boilerplate coding method
# Launch the graph in a session.
# with tf.compat.v1.Session() as ses:
#     # Build a graph.
#     a = tf.constant(5.0)
#     b = tf.constant(6.0)
#     c = a * b
#
#     # Evaluate the tensor `c`.
#     print(ses.run(c))
#     print("Multiplication with constants: %i" % ses.run(a * b))


# to get rid of the above method we use one simple line
# and it is enabled by default
tf.compat.v1.enable_eager_execution()

a = tf.constant([[1, 4], [3, 6]])
b = tf.constant([[2, 7], [8, 5]])
c = a * b

print("Multiplication with constants: ",  c)








