from dataprocess import load_data , points_dataset
from model import Classifier
import numpy as np
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F
import math
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
PATH = 'purning_test.pt'

epoch_number = 700
epsi = 0.000000001

def train(train_data, model, criterion, optimizer):
    loss_counter = 0.0
    for i, (images, labels) in enumerate(train_data):
        images = Variable(images).float()
        labels = Variable(labels).float()
        labels = labels.view(labels.shape[0],-1)
        images = images.view(images.shape[0],-1)
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
            labels = labels.view(labels.shape[0],-1)
            images = images.view(images.shape[0],-1)
            outputs = model(images)
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(images.detach().numpy(), outputs.detach().numpy(), 'o')
            ax2.plot(images.detach().numpy(), labels.detach().numpy(), 'o', )
        plt.show()


# pretty sure this is broken
def sum_of_products_naive(path, L, weights, sum = 0):
    #print(path, L, total)
    if L == 0: # reached the end of the path
        product = 1
        for index, weight in enumerate(path):
            product = product * weights[len(path) - index - 1][weight[1]][weight[0]]
        return sum + product.item()
    else:
        before = path[len(path)-1][1]
        for after in range(len(weights[L-1])):
            sop = sum_of_products_naive(path + [(before, after)], L-1, weights, sum)
        return sop


#returns the coordinates and weight of the smallest weight
def sum_of_products_backwards(weights):
    # names weights <layer weight is reaching towards> <after> <before>
    sops = {}
    for l_index, layer in enumerate(weights):
        for a_index, after in enumerate(layer):
            for b_index, weight in enumerate(after):
                if l_index == 0:
                    sops.update({str(l_index) + str(a_index) + str(b_index) : weight})
                else:
                    sum = 0

                    # add to the sum all the weights which have a before value equal
                    # to this weight's after value
                    for index, _ in enumerate(weights[l_index-1]):
                        sum += sops[str(l_index-1) + str(index) + str(a_index)]
                    
                    # print(str(l_index) + str(a_index) + str(b_index), weight, sum)
                    sops.update({str(l_index) + str(a_index) + str(b_index) : weight * sum})
    return sops['201']

def prune(model):
    weights = []
    for index, layer in enumerate(model.parameters()):
        if index % 2 == 0:
            weights.append(layer)
    weights.reverse() # So that the final layer is layer 0
    print(sum_of_products_backwards(weights))

def main(doPruning = False):
    model = Classifier()
    train_data, test_data = load_data()
    if doPruning:
        model.load_state_dict(torch.load(PATH))
        prune(model)
    else:
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.003)
        old_cost = train(train_data, model, criterion, optimizer)
        for epoch in range(1,epoch_number):
            print(epoch)
            new_cost = train(train_data, model, criterion, optimizer)
            if abs((new_cost - old_cost)) / new_cost  < epsi:
                break
            old_cost = new_cost
        evaluate(test_data, model)
        torch.save(model.state_dict(), PATH)

    
if __name__ == "__main__":
    main(doPruning = False)
    