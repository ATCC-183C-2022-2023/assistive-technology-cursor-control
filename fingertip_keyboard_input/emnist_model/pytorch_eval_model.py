import torch
import pytorch_model_class
import matplotlib.pylab as plt
import data_loader
import sys
from tqdm import tqdm
from pytorch_model_class import DEVICE

def eval(model, train_noise = False, validation_noise = False):
    # load data
    train_data, validation_data = data_loader.load_emnist_data(train_noise = train_noise, validation_noise = validation_noise)
    train_loader, validation_loader = data_loader.create_data_loader(train_data, validation_data)

    # train accuracy
    train_correct = 0
    for x, y in tqdm(train_loader):
        x = x.to(DEVICE)
        y = torch.sub(y, 1)  # in EMNIST-letter split, y is labeled from 1-26, so we subtract 1 to make 0-25
        y = y.to(DEVICE)
        z = model(x)
        _, label = torch.max(z, 1)
        train_correct += (label == y).sum().item()

    train_accuracy = 100 * (train_correct / len(train_data))

    # validation accuracy
    validation_correct = 0
    for x, y in validation_loader:
        x = x.to(DEVICE)
        y = torch.sub(y, 1)  # in EMNIST-letter split, y is labeled from 1-26, so we subtract 1 to make 0-25
        y = y.to(DEVICE)
        z = model(x)
        _, label = torch.max(z, 1)
        validation_correct += (label == y).sum().item()

    validation_accuracy = 100 * (validation_correct / len(validation_data))


    return train_accuracy, validation_accuracy


if __name__ == "__main__":
    
    save_dir = './saved_models'
    # Set the parameters and create the model
    if len(sys.argv) >= 2:
        save_dir = sys.argv[1]

    model = pytorch_model_class.CNN_SRM().to(DEVICE)
    # model = pytorch_model_class.NetReluDeep()

    # load model and evalulate
    model.load_model(path = save_dir)
    train_accuracy, validation_accuracy = eval(model, train_noise = False, validation_noise = False)

    # print the final accuracy
    print('train_accuracy: ', train_accuracy)
    print('validation_accuracy: ', validation_accuracy)