import tensorflow as tf
import timeit
import sys
sys.path.append("../..") 

from Models.KerasTF import dense

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (_, _) = mnist.load_data()
train_images = train_images / 255.0
batch_size = 32
ds_image = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size, drop_remainder=True)
ds_labels = tf.data.Dataset.from_tensor_slices(train_labels).batch(batch_size, drop_remainder=True)

model = dense.DenseModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

SETUP_CODE = '''
from __main__ import model, ds_image, ds_labels, optimizer, grad
import tensorflow as tf
num_epochs = 200'''

TEST_CODE = '''
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in zip(ds_image.as_numpy_iterator(), ds_labels.as_numpy_iterator()):
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg.update_state(loss_value)
        epoch_accuracy.update_state(y, model(x, training=True))

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
'''
training_time = timeit.timeit(setup=SETUP_CODE, stmt=TEST_CODE, number=1)

print("name\ttime\n------------------------")
print("Total time to train\t{:.7f}".format(training_time))
