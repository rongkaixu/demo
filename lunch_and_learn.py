#Step 1: Import TensorFlow and Keras, the high-level API to build and train models in TensorFlow

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
input("Import done! Press Enter to continue...")

#Step 2: Import the specific MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
# Label	Class
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Step 3: Explore the data
train_images.shape
#There are 60,000 images in the training set, with each image represented as 28 x 28 pixels

len(train_labels)
#There are 60,000 corresponding labels in the training set
#Each label is an integer between 0 and 9:
    
len(test_labels)
#And the test set contains 10,000 images labels:

#Step 4: Preprocess the data
#The data must be preprocessed before training the network. If you inspect the first few image
#in the training set, you will see that the pixel values fall in the range of 0 to 255:
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

plt.imshow(train_images[5])
plt.colorbar()
plt.grid(False)
plt.show()

#Scale these values to a range of 0 to 1 before feeding them to the neural network model.
#To do so, divide the values by 255. It's important that the training set and the testing set be preprocessed in the same way:
train_images = train_images / 255.0

test_images = test_images / 255.0

#To verify that the data is in the correct format and that you're ready to build and train the network,
#let's display some images from the training set and display the class name below each image.

plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

#now let's display them    
plt.show()

#Step 5: Build model
#Building the neural network requires configuring the layers of the model, then compiling the model.
#The basic building block of a neural network is the layer. Layers extract representations from the data
#fed into them. Hopefully, these representations are meaningful for the problem at hand.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

#Now we can compile the model now
#model's compile step:
#1. Loss function —This measures how accurate the model is during training. 
#2. Optimizer —This is how the model is updated based on the data it sees and its loss function.
#3. Metrics —Used to monitor the training and testing steps.
#Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#Step 6: Train the model (supervised)
#Training the neural network model requires the following steps:
#1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
#2. The model learns to associate images and labels.
#3. You ask the model to make predictions about a test set—in this example, the test_images array.
#4. Verify that the predictions match the labels from the test_labels array.

model.fit(train_images, train_labels, epochs=10)
#Why we need more than 1 epoch?
#Gradient Descent which is an iterative process. So, updating the weights with single pass or one epoch is not enough.
#Did you see the improving accuracy after each epoch iteration?

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
#With the model trained, you can use it to make predictions about some images. The model's linear outputs, logits.
#Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.

#Step 7: Use the model to test new data
# Grab an image from the test dataset.
img = test_images[1]

plt.imshow(test_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
# Add the image to a batch where it's the only member.

img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = probability_model.predict(img)
#keras.Model.predict returns a list of lists—one list for each image in the batch of data.
#Grab the predictions for our (only) image in the batch:

print(predictions_single)

max_index = np.argmax(predictions_single)
print("Our NN predicts it's a", class_names[max_index])



