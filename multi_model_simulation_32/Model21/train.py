import Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import Data_setup
import time
from torch.utils.data import DataLoader
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
device = 'cpu'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def accuracy(model, dataset, device):
    """
    Compute the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model` - A PyTorch MLPModel
        `dataset` - A data structure that acts like a list of 2-tuples of
                  the form (x, t), where `x` is a PyTorch tensor of shape
                  [400,1] representinga pattern,
                  and `t` is the corresponding binary target label

    Returns: a floating-point value between 0 and 1.
    """

    correct, total = 0, 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    for pattern, t in loader:
        # X = img.reshape(-1, 784)
        pattern = pattern[:,:400].to(device)
        t = t.to(device)
        z = model(pattern)
        y = torch.sigmoid(z)
        pred = (y >= 0.5).int()
        # pred should be a [N, 1] tensor with binary
        # predictions, (0 or 1 in each entry)

        correct += int(torch.sum(t == pred))
        total += t.shape[0]
    # if total == 0:
    #     return 0.0
    return correct / total

if __name__ == "__main__":

    def train_model(model,                # an instance of MLPModel
                    train_data,           # training data
                    val_data,             # validation data
                    learning_rate=0.001,
                    batch_size=64,
                    num_epochs=500,
                    plot_every=50,        # how often (in # iterations) to track metrics
                    plot=True,
                    device= device):           # whether to plot the training curve
    # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                shuffle=True) # reshuffle minibatches every epoch
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters())
        model = model.to(device)

        # these lists will be used to track the training progress
        # and to plot the training curve
        iters, train_loss, train_acc, val_acc = [], [], [], []
        iter_count = 0 # count the number of iterations that has passed

        try:
            start = time.time()
            for e in range(num_epochs):
                for i, (patterns, labels) in enumerate(train_loader):
                    patterns = patterns[:,:400].to(device)

                    # print(patterns[:,:200].shape)
                    labels = labels.to(device)
                    # torch.reshape(labels, (10,1))

                    z = model(patterns).float()
                    # print(z.shape)
                    # print(labels.shape)
                    # break
                    loss = criterion(z, labels.float())
                    

                    loss.backward() # propagate the gradients
                    optimizer.step() # update the parameters
                    optimizer.zero_grad() # clean up accumualted gradients


                    iter_count += 1
                    if iter_count % plot_every == 0:
                        iters.append(iter_count)
                        ta = accuracy(model, train_data, device)
                        va = accuracy(model, val_data, device)
                        train_loss.append(float(loss))
                        train_acc.append(ta)
                        val_acc.append(va)
                        end = time.time()
                        time_taken = round(end - start, 3)
                        print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va, 'Time taken:', time_taken)
        finally:
            if plot:
                plt.figure()
                plt.plot(iters[:len(train_loss)], train_loss)
                plt.title("Loss over iterations")
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.savefig(os.path.join(script_directory, 'training_loss.png'))

                plt.figure()
                plt.plot(iters[:len(train_acc)], train_acc)
                plt.plot(iters[:len(val_acc)], val_acc)
                plt.title("Accuracy over iterations")
                plt.xlabel("Iterations")
                plt.ylabel("Accuracy")
                plt.legend(["Train", "Validation"])
                plt.savefig(os.path.join(script_directory, 'accuracy.png'))

    train_data = Data_setup.train_data
    validation_data = Data_setup.val_data
    test_data = Data_setup.test_data

    model = Model.MLPModel()
    train_model(model, train_data, validation_data)

    test_accuracy = accuracy(model, test_data, device=device)
    print(test_accuracy)


    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    def get_prediction(model, data, sample=1000, device=device):
        loader = torch.utils.data.DataLoader(data, batch_size=sample, shuffle=True)
        for X, t in loader:
            z = model(X[:,:400].to(device))
            y = torch.sigmoid(z)
            break
        y = y.detach().cpu().numpy()
        t = t.detach().cpu().numpy()
        return y, t

    y, t = get_prediction(model, validation_data)
    y = y > 0.5
    cm = confusion_matrix(t, y)
    cmp = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])
    cmp.plot()
    plt.title("Confusion Matrix (Val Data)")
    plt.savefig(os.path.join(script_directory, 'confusion_matrix.png'))

    torch.save(model.state_dict(), os.path.join(script_directory, 'model21.pth'))
    # print(model.state_dict())