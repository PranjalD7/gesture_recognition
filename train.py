import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__
from PIL import Image

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(r'C:/Users/pranj/OneDrive/Desktop/ges_dataset/archive (2)/train/train',
                                                 target_size = (50, 50),
                                                 batch_size = 64,
                                                 class_mode = 'sparse')


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(r'C:/Users/pranj/OneDrive/Desktop/ges_dataset/archive (2)/test/test',
                                            target_size = (50, 50),
                                            batch_size = 64,
                                            class_mode = 'sparse')

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[50, 50, 3],padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))
cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))
cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=1024, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=20, activation='softmax'))

# Part 3 - Training the CNN

# Compiling the CNN

cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs=25)