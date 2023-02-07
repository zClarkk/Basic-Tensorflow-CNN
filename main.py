import tensorflow as tf
import pathlib
import MyDataset
import MyModel

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
img_height = 180
img_width = 180
num_classes = 5

### Basic dataset creation
# train_ds = MyDataset.basic_create_train_ds(data_dir, img_height, img_width, batch_size, split=0.2)
# val_ds = MyDataset.basic_create_val_ds(data_dir, img_height, img_width, batch_size, split=0.2)

### Manual dataset creation
ds = MyDataset.MyDataset(data_dir, img_height, img_width, batch_size)
train_ds, val_ds = ds.manual_create_ds(split=0.2)

# ToDo Dataset from Tensorflow catalog
# train_ds, val_ds, test_ds, num_classes = ds.dataset_from_catalog()

### Optional size check
# print(tf.data.experimental.cardinality(train_ds).numpy())
# print(tf.data.experimental.cardinality(val_ds).numpy())

model = MyModel.create_model(num_classes)
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)
