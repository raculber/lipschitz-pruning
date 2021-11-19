from matplotlib import pyplot as plt
from torch.autograd import Variable
from data_helper import load_data
import torch.nn.utils.prune as pytorch_prune
from models import classifier, lenet5, vgg16, alexnet
from torch import nn, optim
from tqdm import tqdm
import argparse
import torch
import torch.utils.data

CLASSIFIER_PATH = 'pruning_test.pt'
LENET5_PATH = 'trained_lenet5.pt'
VGG16_PATH = 'trained_vgg16.pt'
ALEXNET_PATH = 'trained_alexnet.pt'

epoch_number = 5
percent = 0.01
learning_rate = 0.2
batch_size = 64
epsi = 0.000000001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='lenet5', choices=['vgg16', 'classifier', 'lenet5', 'alexnet'],
                    help='Architecture: vgg16 | classifier | lenet5 | alexnet default: lenet5 ')
parser.add_argument('--batchsize', type=int, default=64, help='Batch Size')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train')
parser.add_argument('--method', default='custom', choices=['custom', 'magnitude'],
                    help='Method used to prune:'
                         'custom | none | '
                         'magnitude default: custom')
parser.add_argument('--iterations', type=int, default=10,
                    help='Number of iterations to prune model')
parser.add_argument('--train', default='false', choices=['false', 'true'],
                    help='Train model initially before pruning: '
                         'true | false')
args = parser.parse_args()


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
        # Freeze pruned weights
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.grad.data = torch.where(param.data.double() == 0., 0., param.grad.data.double())
        optimizer.step()
        loss_counter += loss.data.item()
    return loss_counter


def train_cnn(train_data, model, criterion, optimizer):
    loss_counter = 0.0
    for i, (images, labels) in enumerate(train_data):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # Freeze pruned weights
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.grad.data = torch.where(param.data.double() == 0., 0., param.grad.data.double())
        optimizer.step()
        loss_counter += loss.data.item()
    return loss_counter


def evaluate(test_data, model, pruningMethod, num_iterations):
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_data):
            images = Variable(images).float()
            labels = Variable(labels).float()
            labels = labels.view(labels.shape[0], -1)
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
            ax1.plot(images.detach().numpy(), outputs.detach().numpy(), 'o')
            ax2.plot(images.detach().numpy(), labels.detach().numpy(), 'o')

        # Turn off tick labels
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])

        ax1.title.set_text('Model Output')
        ax2.title.set_text('Ground Truth')

        plt.text(-0.225, 0.7,
                 'Results \n' +
                 'Method: ' + pruningMethod + '\n' +
                 'num epochs: ' + str(args.epochs) + '\n' +
                 'pruning iterations: ' + str(num_iterations) + '\n' +
                 'initial pruning rate: ' + str(percent) + '\n' +
                 'learning rate: ' + str(learning_rate) + '\n' +
                 'batch size: ' + str(batch_size) + '\n' +
                 'epsi: ' + str(epsi),
                 fontsize=10, weight='bold')
        plt.show()


def evaluate_cnn(test_data, model):
    with torch.no_grad():
        total = 0
        correct = 0
        accuracy = 0.0
        for i, (images, labels) in enumerate(test_data):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            value, predicted = torch.max(outputs, 1)
            total += outputs.size(0)
            correct += torch.sum(predicted == labels)
            accuracy = correct * 100.0 / total
        print(accuracy)


def training_loop(train_data, test_data, model, criterion, num_iterations):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    if args.arch == "lenet5" or args.arch == "vgg16" or args.arch == "alexnet":
        old_cost = train_cnn(train_data, model, criterion, optimizer)
    else:
        old_cost = train(train_data, model, criterion, optimizer)
    for _ in tqdm(range(1, args.epoch_number)):
        if args.arch == "lenet5" or args.arch == "vgg16" or args.arch == "alexnet":
            new_cost = train_cnn(train_data, model, criterion, optimizer)
        else:
            new_cost = train(train_data, model, criterion, optimizer)
        if abs((new_cost - old_cost)) / new_cost < epsi:
            break
        old_cost = new_cost
    if args.arch == "lenet5" or args.arch == "vgg16" or args.arch == "alexnet":
        evaluate_cnn(test_data, model)
    else:
        evaluate(test_data, model, pruningMethod=args.method, num_iterations=num_iterations)


def sum_of_products_backwards(weights):
    # names weights <layer weight is reaching towards> <after> <before>
    sops = {}
    for l_index, layer in enumerate(weights):
        for a_index, after in enumerate(layer):
            for b_index, weight in enumerate(after):
                weight = abs(weight)
                if l_index == 3:
                    print(l_index)
                    print(weight.shape)
                    print(weight)
                if l_index == 4:
                    print(l_index)
                    print(weight.shape)
                    print(weight)
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
    for name, layer in model.named_parameters():
        if 'weight' in name and 'fc' in name:
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
    with torch.no_grad():
        for i in prune_indices:
            parsed_index = i.split(',')
            parsed_index = [int(i) for i in parsed_index]
            weights[parsed_index[0]][parsed_index[1]][parsed_index[2]] = 0  # set weight to 0


def main():
    num_iterations = args.iterations
    if args.arch == "lenet5":
        model = lenet5.LeNet5(num_classes=10)
        path = LENET5_PATH
        criterion = nn.CrossEntropyLoss()
    elif args.arch == "vgg16":
        model = vgg16.VGG16(num_classes=10)
        path = VGG16_PATH
        criterion = nn.CrossEntropyLoss()
    elif args.arch == "alexnet":
        model = alexnet.AlexNet(num_classes=10)
        path = ALEXNET_PATH
        criterion = nn.CrossEntropyLoss()
    else:
        path = CLASSIFIER_PATH
        model = classifier.Classifier()
        criterion = nn.MSELoss()
    train_data, test_data = load_data(args)
    if args.train == 'true':
        training_loop(train_data, test_data, model=model, criterion=criterion,
                      num_iterations=num_iterations)
    model.load_state_dict(torch.load(path))
    if args.method == 'magnitude':
        for i in range(1, num_iterations + 1):
            prune_rate = (percent ** (1 / i))
            for module in model.named_modules():
                if isinstance(module[1], nn.Linear):
                    pytorch_prune.l1_unstructured(module[1], 'weight', prune_rate)
                    pytorch_prune.remove(module[1], 'weight')
            print('Iteration ' + str(i))
            print("Prune rate " + str(prune_rate))
            print('Zero weights ' + str(count_zero_weights(model)))
            training_loop(train_data, test_data, model=model, criterion=criterion,
                          num_iterations=num_iterations)
            print('Zero weights After training ' + str(count_zero_weights(model)))
    elif args.method == 'custom':
        for i in range(1, num_iterations + 1):
            prune_rate = (percent ** (1 / i))
            prune(model, prune_rate)
            print('Iteration ' + str(i))
            print("Prune rate " + str(prune_rate))
            print('Zero weights ' + str(count_zero_weights(model)))
            training_loop(train_data, test_data, model=model, criterion=criterion,
                          num_iterations=num_iterations)
            print('Zero Weights After training ' + str(count_zero_weights(model)))
            print("Total Weights " + str(count_total_weights(model)))
    torch.save(model.state_dict(), path)


def count_zero_weights(model):
    zeros = 0
    for param in model.parameters():
        if param is not None:
            zeros += torch.sum((param == 0).int()).data.item()
    return zeros


def count_total_weights(model):
    weight_sum = 0
    for name, layer in model.named_parameters():
        if not layer.requires_grad:
            num_weights = layer.numel()
            weight_sum += num_weights
    return weight_sum


if __name__ == "__main__":
    main()
