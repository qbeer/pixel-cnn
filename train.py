import tensorflow as tf

import tensorflow_datasets as tfds
from model import PixelCNN

ds_train = tfds.load('cifar10',
                     split='train',
                     shuffle_files='True',
                     batch_size=16,
                     as_supervised=True)
ds_test = tfds.load('cifar10',
                    split='test',
                    shuffle_files='False',
                    batch_size=32,
                    as_supervised=True)

model = PixelCNN()

def neg_log_likelihood(target, output):
    

for images, labels in ds_train.take(1):
    print(images)
