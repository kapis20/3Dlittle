import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

########################
# paths 
########################
train_csv_path = "/home/kapis20/Projects/3D_new/3Dlittle/data/train/_annotations.csv"
valid_csv_path = "/home/kapis20/Projects/3D_new/3Dlittle/data/valid/_annotations.csv"
test_csv_path = "/home/kapis20/Projects/3D_new/3Dlittle/data/test/_annotations.csv"


spec = model_spec.get('efficientdet_lite1')

train_data = object_detector.DataLoader.from_csv(train_csv_path)
validation_data = object_detector.DataLoader.from_csv(valid_csv_path)
test_data = object_detector.DataLoader.from_csv(test_csv_path)

#print(type(train_data))
print(type(train_data))  # Check the type of train_data
print(train_data)  # Check the content of train_data


# model = object_detector.create(
#     train_data=train_data,
#     model_spec=spec,
#     validation_data=validation_data,
#     epochs=10,
# )


# model = object_detector.create(
#     train_data=train_data,
#     model_spec=spec,
#     validation_data=validation_data,
#     epochs=1,
# )
# # Evaluate the model
# model.evaluate(test_data)
