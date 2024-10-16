import numpy as np


def create_data(z,n, on=True):
    """
    Create an n-dimensional vector where each element is either 0 or 1 randomly.
    The zth element is set to 1 if on=True and 0 if on=False.

    Parameters:
    z (int): The index (1-based) to set specifically.
    n (int): The size of the vector.
    on (bool): If True, set the zth element to 1. If False, set it to 0.

    Returns:
    np.ndarray: The resulting n-dimensional vector.
    """
    # Create a random vector of 0s and 1s
    data = np.random.randint(2,size=n)
    # Set the zth element to the desired value (convert z from 1-based to 0-based index)
    data[z - 1] = 1 if on else 0
    return data


def create_data_sets(n, samples_per_set):
    data_sets = []

    for z in range(1, n + 1):
        data1 = [create_data(z, n, True) for _ in range(samples_per_set)]
        label1 = np.ones(samples_per_set)
        data__1 = list(zip(data1, label1))

        data2 = [create_data(z, n, False) for _ in range(samples_per_set)]
        label2 = np.zeros(samples_per_set)
        data__2 = list(zip(data2, label2))

        data_set = data__1 + data__2
        data_sets.append(data_set)

    return data_sets


n = 32
samples_per_set = 2000
data_sets = create_data_sets(n, samples_per_set)

# Unpack the data sets
# print(data_sets)
data_set_1, data_set_2, data_set_3, data_set_4, data_set_5, data_set_6, data_set_7, data_set_8, data_set_9, data_set_10, data_set_11, data_set_12, data_set_13, data_set_14, data_set_15, data_set_16, data_set_17, data_set_18, data_set_19, data_set_20, data_set_21, data_set_22, data_set_23, data_set_24, data_set_25, data_set_26, data_set_27, data_set_28, data_set_29, data_set_30, data_set_31, data_set_32 = data_sets

for data_set in data_sets:
    np.random.shuffle(data_set)

# print(data_set_1[:10])
# np.random.shuffle(data_set_1)
# print(data_set_1[:10])
# np.random.shuffle(data_set_2)
# np.random.shuffle(data_set_3)
# np.random.shuffle(data_set_4)
# np.random.shuffle(data_set_5)
# np.random.shuffle(data_set_6)
# np.random.shuffle(data_set_7)
# np.random.shuffle(data_set_8)
# np.random.shuffle(data_set_9)
# np.random.shuffle(data_set_10)
# print(data_set_4)


# print(data_set_1[0])