import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(r'C:/Users/pranj/OneDrive/Desktop/ges_dataset/archive (2)/train/train',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(r'C:/Users/pranj/OneDrive/Desktop/ges_dataset/archive (2)/test/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

