import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np


AUTOTUNE = tf.data.AUTOTUNE

def basic_create_train_ds(data_dir, img_height, img_width, batch_size, split):
  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=split,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  return train_ds

def basic_create_val_ds(data_dir, img_height, img_width, batch_size, split):
  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=split,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  return val_ds

class MyDataset:
  def __init__(self, data_dir, img_height, img_width, batch_size):
    self.data_dir = data_dir
    self.img_height = img_height
    self.img_width = img_width
    self.batch_size = batch_size
    self.class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

  def ds_list(self):
    image_count = len(list(self.data_dir.glob('*/*.jpg')))
    list_ds = tf.data.Dataset.list_files(str(self.data_dir / '*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    return image_count, list_ds

  def configure_for_performance(self, ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(self.batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

  def get_label(self, file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == self.class_names
    return tf.argmax(one_hot)

  def decode_img(self, img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [self.img_height, self.img_width])

  def process_path(self, file_path):
    label = self.get_label(file_path)
    img = tf.io.read_file(file_path)
    img = self.decode_img(img)
    return img, label

  def manual_create_ds(self, split):
    image_count, list_ds = self.ds_list()
    val_size = int(image_count * split)

    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    train_ds = train_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)

    train_ds = self.configure_for_performance(train_ds)
    val_ds = self.configure_for_performance(val_ds)

    return train_ds, val_ds

  # ToDo
  # def dataset_from_catalog(self):
  #   (train_ds, val_ds, test_ds), metadata = tfds.load(
  #     'tf_flowers',
  #     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
  #     with_info=True,
  #     as_supervised=True,
  #   )
  #   num_classes = metadata.features['label'].num_classes
  #   train_ds = self.configure_for_performance(train_ds)
  #   val_ds = self.configure_for_performance(val_ds)
  #   test_ds = self.configure_for_performance(test_ds)
  #
  #   return train_ds, val_ds, test_ds, num_classes
