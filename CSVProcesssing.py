# import pandas as pd

# # Define paths to your CSV files
# train_csv_path = "/home/kapis20/Projects/3D_new/3Dlittle/data/train/_annotations.csv"
# valid_csv_path = "/home/kapis20/Projects/3D_new/3Dlittle/data/valid/_annotations.csv"
# test_csv_path = "/home/kapis20/Projects/3D_new/3Dlittle/data/test/_annotations.csv"

# # Define a function to add the set column
# def add_set_column(csv_path, set_name):
#     try:
#         # Load the CSV file
#         df = pd.read_csv(csv_path)
        
#         # Add the set column as the first column
#         df.insert(0, 'set_name', set_name)
        
#         # Save the modified CSV back
#         df.to_csv(csv_path, index=False)
#         print(f"Successfully added '{set_name}' column to {csv_path}")
#     except Exception as e:
#         print(f"Error processing {csv_path}: {e}")

# # Add the column to each CSV
# add_set_column(train_csv_path, 'TRAINING')
# add_set_column(valid_csv_path, 'VALIDATION')
# add_set_column(test_csv_path, 'TEST')


import pandas as pd

# # Define your CSV file path and image directory
csv_path = "/home/kapis20/Projects/3D_new/3Dlittle/data/train/_annotations.csv"
# image_dir = "/home/kapis20/Projects/3D_new/3Dlittle/data/train/"  # Image directory

# # Load the CSV file
# df = pd.read_csv(csv_path)

# # Assuming the image names are in the second column (index 1) of the CSV
# df.iloc[:, 1] = image_dir + df.iloc[:, 1]  # Prepend the path to the image names

# # Save the updated CSV
# df.to_csv(csv_path, index=False)

# print("Updated CSV with full paths in the second column.")

# from tflite_model_maker import object_detector

# help(object_detector.DataLoader.from_csv)

# Load the CSV
#df = pd.read_csv(csv_path)

# # Drop columns by index (2 and 3)
# df.drop(df.columns[[2, 3]], axis=1, inplace=True)

# # Save the updated CSV back to the file
# df.to_csv(csv_path, index=False)

# print("Removed columns 2 and 3 successfully.")


df = pd.read_csv(csv_path)

# Load the CSV
df = pd.read_csv(csv_path)

# Get the name of the last column
last_column = df.columns[-1]

# Create a new column order: Insert the last column at index 2 (3rd position)
new_column_order = list(df.columns[:2]) + [last_column] + list(df.columns[2:-1])

# Reorder the DataFrame
df = df[new_column_order]

# Save the updated CSV back to the file
df.to_csv(csv_path, index=False)

print(f"Moved last column '{last_column}' to the 3rd position.")