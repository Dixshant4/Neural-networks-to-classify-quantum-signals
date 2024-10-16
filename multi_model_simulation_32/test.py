import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from Model1.Model import MLPModel
from Model1.get_power_readings import transformed_dataset, dataset
# from Model1.Data_setup import test_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Append the directories to the system path
sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model1')
model1_path = '/home/dixshant/Multi_model_sim_32_d/Model1/model1.pth'
model1 = MLPModel()
model1.load_state_dict(torch.load(model1_path))
model1.to(device)
model1.eval()
# print(f'model1: {model1.state_dict()}')

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model2')
model2_path = '/home/dixshant/Multi_model_sim_32_d/Model2/model2.pth'
model2 = MLPModel()
model2.load_state_dict(torch.load(model2_path))
model2.to(device)
model2.eval()
# print(f'model2: {model2.state_dict()}')


sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model3')
model3_path = '/home/dixshant/Multi_model_sim_32_d/Model3/model3.pth'
model3 = MLPModel()
model3.load_state_dict(torch.load(model3_path))
model3.to(device)
model3.eval()
# print(f'model3: {model3.state_dict()}')

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model4')
model4_path = '/home/dixshant/Multi_model_sim_32_d/Model4/model4.pth'
model4 = MLPModel()
model4.load_state_dict(torch.load(model4_path))
model4.to(device)
model4.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model5')
model5_path = '/home/dixshant/Multi_model_sim_32_d/Model5/model5.pth'
model5 = MLPModel()
model5.load_state_dict(torch.load(model5_path))
model5.to(device)
model5.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model6')
model6_path = '/home/dixshant/Multi_model_sim_32_d/Model6/model6.pth'
model6 = MLPModel()
model6.load_state_dict(torch.load(model6_path))
model6.to(device)
model6.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model7')
model7_path = '/home/dixshant/Multi_model_sim_32_d/Model7/model7.pth'
model7 = MLPModel()
model7.load_state_dict(torch.load(model7_path))
model7.to(device)
model7.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model8')
model8_path = '/home/dixshant/Multi_model_sim_32_d/Model8/model8.pth'
model8 = MLPModel()
model8.load_state_dict(torch.load(model8_path))
model8.to(device)
model8.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model9')
model9_path = '/home/dixshant/Multi_model_sim_32_d/Model9/model9.pth'
model9 = MLPModel()
model9.load_state_dict(torch.load(model9_path))
model9.to(device)
model9.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model10')
model10_path = '/home/dixshant/Multi_model_sim_32_d/Model10/model10.pth'
model10 = MLPModel()
model10.load_state_dict(torch.load(model10_path))
model10.to(device)
model10.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model11')
model11_path = '/home/dixshant/Multi_model_sim_32_d/Model11/model11.pth'
model11 = MLPModel()
model11.load_state_dict(torch.load(model11_path))
model11.to(device)
model11.eval()
# print(f'model1: {model1.state_dict()}')

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model12')
model12_path = '/home/dixshant/Multi_model_sim_32_d/Model12/model12.pth'
model12 = MLPModel()
model12.load_state_dict(torch.load(model12_path))
model12.to(device)
model12.eval()
# print(f'model2: {model2.state_dict()}')


sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model13')
model13_path = '/home/dixshant/Multi_model_sim_32_d/Model13/model13.pth'
model13 = MLPModel()
model13.load_state_dict(torch.load(model13_path))
model13.to(device)
model13.eval()
# print(f'model3: {model3.state_dict()}')

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model14')
model14_path = '/home/dixshant/Multi_model_sim_32_d/Model14/model14.pth'
model14 = MLPModel()
model14.load_state_dict(torch.load(model14_path))
model14.to(device)
model14.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model15')
model15_path = '/home/dixshant/Multi_model_sim_32_d/Model15/model15.pth'
model15 = MLPModel()
model15.load_state_dict(torch.load(model15_path))
model15.to(device)
model15.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model16')
model16_path = '/home/dixshant/Multi_model_sim_32_d/Model16/model16.pth'
model16 = MLPModel()
model16.load_state_dict(torch.load(model16_path))
model16.to(device)
model16.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model17')
model17_path = '/home/dixshant/Multi_model_sim_32_d/Model17/model17.pth'
model17 = MLPModel()
model17.load_state_dict(torch.load(model17_path))
model17.to(device)
model17.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model18')
model18_path = '/home/dixshant/Multi_model_sim_32_d/Model18/model18.pth'
model18 = MLPModel()
model18.load_state_dict(torch.load(model18_path))
model18.to(device)
model18.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model19')
model19_path = '/home/dixshant/Multi_model_sim_32_d/Model19/model19.pth'
model19 = MLPModel()
model19.load_state_dict(torch.load(model19_path))
model19.to(device)
model19.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model20')
model20_path = '/home/dixshant/Multi_model_sim_32_d/Model20/model20.pth'
model20 = MLPModel()
model20.load_state_dict(torch.load(model20_path))
model20.to(device)
model20.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model21')
model21_path = '/home/dixshant/Multi_model_sim_32_d/Model21/model21.pth'
model21 = MLPModel()
model21.load_state_dict(torch.load(model21_path))
model21.to(device)
model21.eval()
# print(f'model1: {model1.state_dict()}')

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model22')
model22_path = '/home/dixshant/Multi_model_sim_32_d/Model22/model22.pth'
model22 = MLPModel()
model22.load_state_dict(torch.load(model22_path))
model22.to(device)
model22.eval()
# print(f'model2: {model2.state_dict()}')


sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model23')
model23_path = '/home/dixshant/Multi_model_sim_32_d/Model23/model23.pth'
model23 = MLPModel()
model23.load_state_dict(torch.load(model23_path))
model23.to(device)
model23.eval()
# print(f'model3: {model3.state_dict()}')

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model24')
model24_path = '/home/dixshant/Multi_model_sim_32_d/Model24/model24.pth'
model24 = MLPModel()
model24.load_state_dict(torch.load(model24_path))
model24.to(device)
model24.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model25')
model25_path = '/home/dixshant/Multi_model_sim_32_d/Model25/model25.pth'
model25 = MLPModel()
model25.load_state_dict(torch.load(model25_path))
model25.to(device)
model25.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model26')
model26_path = '/home/dixshant/Multi_model_sim_32_d/Model26/model26.pth'
model26 = MLPModel()
model26.load_state_dict(torch.load(model26_path))
model26.to(device)
model26.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model27')
model27_path = '/home/dixshant/Multi_model_sim_32_d/Model27/model27.pth'
model27 = MLPModel()
model27.load_state_dict(torch.load(model27_path))
model27.to(device)
model27.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model28')
model28_path = '/home/dixshant/Multi_model_sim_32_d/Model28/model28.pth'
model28 = MLPModel()
model28.load_state_dict(torch.load(model28_path))
model28.to(device)
model28.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model29')
model29_path = '/home/dixshant/Multi_model_sim_32_d/Model29/model29.pth'
model29 = MLPModel()
model29.load_state_dict(torch.load(model29_path))
model29.to(device)
model29.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model30')
model30_path = '/home/dixshant/Multi_model_sim_32_d/Model30/model30.pth'
model30 = MLPModel()
model30.load_state_dict(torch.load(model30_path))
model30.to(device)
model30.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model31')
model31_path = '/home/dixshant/Multi_model_sim_32_d/Model31/model31.pth'
model31 = MLPModel()
model31.load_state_dict(torch.load(model31_path))
model31.to(device)
model31.eval()

sys.path.append('/home/dixshant/Multi_model_sim_32_d/Model32')
model32_path = '/home/dixshant/Multi_model_sim_32_d/Model32/model32.pth'
model32 = MLPModel()
model32.load_state_dict(torch.load(model32_path))
model32.to(device)
model32.eval()

raw_power = np.array([item[0] for item in transformed_dataset], dtype=np.float32)
power = torch.from_numpy(raw_power)

raw_image = np.array([item[0] for item in dataset], dtype=np.float32)
image = torch.from_numpy(raw_image)

# Custom accuracy function
def accuracy(models, transformed_dataset, raw_image, device, n):
    correct, total = 0, 0
    i = 0
    for pattern in transformed_dataset:
        pattern = pattern.to(device)
        # raw_image[i].to(device)

        preds = torch.zeros(n)
        for count, model in enumerate(models):
           preds[count] = torch.sigmoid(model(pattern)).round().int()
           
        if torch.equal(preds.to(device), raw_image[i].to(device)):
         correct += 1
        i += 1
    total += len(transformed_dataset)
    
    return correct / total

# # # Calculate accuracy
models = (model1.to(device), model2.to(device), model3.to(device), model4.to(device), model5.to(device), model6.to(device), model7.to(device), model8.to(device), model9.to(device), model10.to(device), model11.to(device), model12.to(device), model13.to(device), model14.to(device), model15.to(device), model16.to(device), model17.to(device), model18.to(device), model19.to(device), model20.to(device), model21.to(device), model22.to(device), model23.to(device), model24.to(device), model25.to(device), model26.to(device), model27.to(device), model28.to(device), model29.to(device), model30.to(device), model31.to(device), model32.to(device))
acc = accuracy(models, power, image, device, n=32)
print(f'Accuracy: {acc * 100:.2f}%')

# def accuracy(model, dataset, device):
#     """
#     Compute the accuracy of `model` over the `dataset`.
#     We will take the **most probable class**
#     as the class predicted by the model.

#     Parameters:
#         `model` - A PyTorch MLPModel
#         `dataset` - A data structure that acts like a list of 2-tuples of
#                   the form (x, t), where `x` is a PyTorch tensor of shape
#                   [400,1] representinga pattern,
#                   and `t` is the corresponding binary target label

#     Returns: a floating-point value between 0 and 1.
#     """

#     correct, total = 0, 0
#     loader = torch.utils.data.DataLoader(dataset, batch_size=10)
#     for pattern, t in loader:
#         # X = img.reshape(-1, 784)
#         pattern = pattern[:,:400].to(device)
#         t = t.to(device)
#         z = model(pattern)
#         y = torch.sigmoid(z)
#         pred = (y >= 0.5).int()
#         # pred should be a [N, 1] tensor with binary
#         # predictions, (0 or 1 in each entry)

#         correct += int(torch.sum(t == pred))
#         total += t.shape[0]
#     # if total == 0:
#     #     return 0.0
#     return correct / total

# print(f"test acc to see model 1: {accuracy(model3, test_data, device)}")