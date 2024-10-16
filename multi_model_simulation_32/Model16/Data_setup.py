import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from get_power_readings import transformed_dataset

# the class which helps create dataset objects
class CSDataset(Dataset):
    def __init__(self, data, labels):
        """
        images: A tensor containing the image data.
        labels: A tensor containing the corresponding labels.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Fetches the sample at index `idx` from the dataset."""
        return self.data[idx], self.labels[idx]

loaded_array = transformed_dataset
# print(loaded_array)
total_size = len(loaded_array)

split1_size = int(total_size * 0.6)
split2_size = int(total_size * 0.2)

# seperating raw data into power and labels
raw_power = np.array([item[0] for item in loaded_array], dtype=np.float32)

raw_labels = np.array([item[1] for item in loaded_array], dtype=np.int64)

# convert np arrays into pytorch tensors
power = torch.from_numpy(raw_power) # each row is a data point
labels = torch.from_numpy(raw_labels)
labels = labels.reshape(len(labels),1)

# labels = labels.reshape(len(labels),1)
# print(labels)

# split the data
train_power = power[:split1_size]
val_power = power[split1_size:split2_size + split1_size]
test_power = power[split1_size + split2_size:]

train_labels = labels[:split1_size]
val_labels = labels[split1_size:split2_size + split1_size]
test_labels = labels[split1_size + split2_size:]

# converting data into dataset objects to pass into Dataloader
train_data = CSDataset(train_power, train_labels)
test_data = CSDataset(test_power, test_labels)
val_data = CSDataset(val_power, val_labels)


train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=10)
val_dataloader = DataLoader(val_data, batch_size=10)


# #
# for x, t in test_data:
#     print(x, t)
#     print(t.shape)
#     break
    
# print(len(train_labels), len(val_labels), len(test_data))
# count_0 = 0
# count_1 = 0
# for pattern, label in test_data:
#     if label==0:
#         count_0 += 1
#     else:
#         count_1 += 1
#
# print(count_0)
# print(count_1)