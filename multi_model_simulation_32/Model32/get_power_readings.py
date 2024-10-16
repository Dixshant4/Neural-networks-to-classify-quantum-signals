import numpy as np
import sys

sys.path.append('/home/dixshant/Multi_model_sim_32_d')
from create_sensing_matrix import sensing_matrix
import create_simulated_images

sensing_matrix = sensing_matrix(32,32)

def transform_dataset(dataset, matrix):
    transformed_dataset = []
    for vector, label in dataset:
        transformed_vector = np.dot(matrix, vector)  # Matrix-vector multiplication
        transformed_dataset.append((transformed_vector, label))
    return transformed_dataset

# Transform the dataset
dataset = create_simulated_images.data_set_32
transformed_dataset = transform_dataset(dataset, sensing_matrix)

# print(transformed_dataset)