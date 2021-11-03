from matplotlib import pyplot as plt
from torch.autograd import Variable

from dataprocess import load_data
from model import Classifier
from torch import nn, optim
import torch.nn.utils.prune as prune
import torch
import torchvision.transforms as transforms
import torch.utils.data


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

def sum_of_products_backwards(weights):
    # names weights <layer weight is reaching towards> <after> <before>
    sops = {}
    for l_index, layer in enumerate(weights):
        for a_index, after in enumerate(layer):
            for b_index, weight in enumerate(after):
                if l_index == 0:
                    sops.update({str(l_index) + ',' + str(a_index) + ',' + str(b_index): weight})
                else:
                    sum = 0

                    # add to the sum all the weights which have a before value equal
                    # to this weight's after value
                    for index, _ in enumerate(weights[l_index - 1]):
                        sum += sops[str(l_index - 1) + ',' + str(index) + ',' + str(a_index)]
                    # print(str(l_index) + str(a_index) + str(b_index), weight, sum)
                    sops.update({str(l_index) + ',' + str(a_index) + ',' + str(b_index): weight * sum})
    sops = dict(sorted(sops.items(), key=lambda item: item[1]))
    return sops


def prune(model, prune_rate):
    weights = []
    for index, layer in enumerate(model.parameters()):
        if index % 2 == 0:
            weights.append(layer)
    weights.reverse()  # So that the final layer is layer 0
    # key: coordinates of weight, value: sum of products
    sop_dict = sum_of_products_backwards(weights)
    # removing pruned weights from our dictionary
    sop_dict = {key: val for key, val in sop_dict.items() if 
            weights[int(key.split(',')[0])][int(key.split(',')[1])][int(key.split(',')[2])] != 0}
        

    

    # Calculate the end index to prune
    prune_end_index = int(prune_rate * len(sop_dict))
    prune_indices = list(sop_dict)[:prune_end_index]

    print('weights remaining:', len(sop_dict))
    print('weights to prune:', len(prune_indices))
    
    with torch.no_grad():
        # this is a good lead: https://www.py4u.net/discuss/254455
        for i in prune_indices:
            parsed_index = i.split(',')
            parsed_index = [int(i) for i in parsed_index]
            print(weights[parsed_index[0]][parsed_index[1]][parsed_index[2]])            
            weights[parsed_index[0]][parsed_index[1]][parsed_index[2]] = 0 # set weight to 0
            weights[parsed_index[0]][parsed_index[1]][parsed_index[2]].requires_grad = False   # freeze weight
            print(weights[parsed_index[0]][parsed_index[1]][parsed_index[2]])
        
def main():
    model = Classifier()
    train_data, test_data = load_data()
    for i in range(1, num_iterations + 1):
        prune_rate = (percent ** (1 / i))
        prune(model, prune_rate)
        print('Iteration ' + str(i))
        print('Prune rate ' + str(prune_rate))
        print('Zero weights before fine tuning ' + str(count_zero_weights(model)))
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        old_cost = train(train_data, model, criterion, optimizer)
        for epoch in range(1, epoch_number):
            new_cost = train(train_data, model, criterion, optimizer)
            if abs((new_cost - old_cost)) / new_cost < epsi:
                break
            old_cost = new_cost
        print('Zero weights after fine tuning ' + str(count_zero_weights(model)))
    evaluate(test_data, model)
    torch.save(model.state_dict(), PATH)

def count_zero_weights(model):
    zeros = 0
    for param in model.parameters():
        if param is not None:
            zeros += torch.sum((param == 0).int()).data.item()
    return zeros


if __name__ == "__main__":
    main()
