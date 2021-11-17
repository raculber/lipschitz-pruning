from matplotlib import pyplot as plt
from torch.autograd import Variable

from dataprocess import load_data
from model import Classifier
from torch import nn, optim
from torch.nn.utils import prune
import torch
import torchvision.transforms as transforms
import torch.utils.data
from pruningMethod import myFavoritePruningMethod
import numpy as np

PATH = 'purning_test.pt'

epoch_number = 300
num_iterations = 3
percent = 0.01
learning_rate = 0.03
batch_size = 64
epsi = 0.000000001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def train(train_data, model, criterion, optimizer):
    loss_counter = 0.0
    for i, (images, labels) in enumerate(train_data):
        images = Variable(images).float()
        labels = Variable(labels).float()
        labels = labels.view(labels.shape[0], -1)
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_counter += loss.data.item()
    return loss_counter


def evaluate(test_data, model):
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_data):
            images = Variable(images).float()
            labels = Variable(labels).float()
            labels = labels.view(labels.shape[0], -1)
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(images.detach().numpy(), outputs.detach().numpy(), 'o')
            ax2.plot(images.detach().numpy(), labels.detach().numpy(), 'o', )
        plt.show()


def sum_of_products_backwards(weights):
    # names weights <layer weight is reaching towards> <after> <before> <flat index>
    sops = {}
    toFlat = {}  # Key : 3d indecies, value : flat indecies
    flat_index = -1
    for l_index, layer in enumerate(weights):
        for a_index, after in enumerate(layer):
            for b_index, weight in enumerate(after):
                flat_index += 1
                if l_index == 0:
                    sops.update({str(l_index) + ',' + str(a_index) + ',' + str(b_index): weight})
                    toFlat.update({str(l_index) + ',' + str(a_index) + ',' + str(b_index): flat_index})
                else:
                    sum = 0

                    # add to the sum all the weights which have a before value equal
                    # to this weight's after value
                    for index, _ in enumerate(weights[l_index - 1]):
                        sum += sops[str(l_index - 1) + ',' + str(index) + ',' + str(a_index)]
                    # print(str(l_index) + str(a_index) + str(b_index), weight, sum)
                    sops.update({str(l_index) + ',' + str(a_index) + ',' + str(b_index): weight * sum})
                    toFlat.update({str(l_index) + ',' + str(a_index) + ',' + str(b_index): flat_index})

    sops = dict(sorted(sops.items(), key=lambda item: item[1]))
    return sops, toFlat


def pruneModel(model, prune_rate):
    parametersToPrune = (
        (model.layers[0], 'weight'),
        (model.layers[2], 'weight'),
        (model.layers[4], 'weight'),
        (model.layers[6], 'weight'),
        (model.layers[8], 'weight'),
        (model.layers[10], 'weight'),
        (model.layers[12], 'weight'),
        (model.layers[14], 'weight'),
        (model.layers[16], 'weight')
    )

    weights = []
    for index, layer in enumerate(model.parameters()):
        if index % 2 == 0:
            weights.append(layer)

    weights.reverse()  # So that the final layer is layer 0

    sop_dict, toFlat = sum_of_products_backwards(weights)

    # Calculate the end index to prune
    prune_end_index = int(prune_rate * len(sop_dict))
    prune_indices = list(sop_dict)[:prune_end_index]
    prune_indices = [toFlat[_] for _ in prune_indices]

    mask = torch.ones(21030)  # TODO: Un-hardcode this value

    for idx in prune_indices:
        mask[idx] = 0

    torch.nn.utils.prune.global_unstructured(
        parametersToPrune,
        pruning_method=myFavoritePruningMethod, Mask=mask  # TODO: move mask calculation to this function
    )

    for parameter in parametersToPrune:
        prune.remove(parameter[0], parameter[1])

    # for index, layer in enumerate(model.layers):
    #     if index % 2 == 0:
    #         print(layer.weight)


def main():
    model = Classifier()
    prune_rate = .1
    train_data, test_data = load_data()
    pruneModel(model, prune_rate)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    old_cost = train(train_data, model, criterion, optimizer)
    for epoch in range(1, epoch_number):
        new_cost = train(train_data, model, criterion, optimizer)
        if abs((new_cost - old_cost)) / new_cost < epsi:
            break
        old_cost = new_cost
    evaluate(test_data, model)
    torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    main()
