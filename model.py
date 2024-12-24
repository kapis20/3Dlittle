import numpy as np
import os
import pandas as pd

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
train_csv_path = "/home/kapis20/Projects/3D_new/3Dlittle/data/train/_annotations_copy.csv"
valid_csv_path = "/home/kapis20/Projects/3D_new/3Dlittle/data/valid/_annotations.csv"
test_csv_path = "/home/kapis20/Projects/3D_new/3Dlittle/data/test/_annotations.csv"




# data = pd.read_csv(train_csv_path)
# print("Columns in the CSV file:")
# print(data.columns)

# for path in [train_csv_path, valid_csv_path, test_csv_path]:
#     if os.path.exists(path):
#         print(f"File exists: {path}")
#     else:
#         print(f"File does NOT exist: {path}")

# for path in [train_csv_path, valid_csv_path, test_csv_path]:
#     if os.access(path, os.R_OK):
#         print(f"File is readable: {path}")
#     else:
#         print(f"File is NOT readable: {path}")

# for path in [train_csv_path, valid_csv_path, test_csv_path]:
#     data = pd.read_csv(path)
#     all_files_exist = all(os.path.exists(file_path) for file_path in data['file_path_column'])
#     if all_files_exist:
#         print(f"All file paths in {path} are valid.")
#     else:
#         print(f"Some file paths in {path} are invalid.")


spec = model_spec.get('efficientdet_lite1')

train_data = object_detector.DataLoader.from_csv(train_csv_path)
#validation_data = object_detector.DataLoader.from_csv(valid_csv_path)
#test_data = object_detector.DataLoader.from_csv(test_csv_path)

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
