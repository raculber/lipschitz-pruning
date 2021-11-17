from matplotlib import pyplot as plt
from torch.autograd import Variable
from dataprocess import load_data
from models import classifier, lenet5, vgg16
from torch import nn, optim
from torchvision import datasets
from tqdm import tqdm
import argparse
import torch
import torchvision.transforms as transforms
import torch.utils.data

PATH = 'purning_test.pt'
LENET5_PATH = 'trained_lenet5.pt'
VGG16_PATH = 'trained_vgg16.pt'
epoch_number = 100
num_iterations = 6
percent = 0.01
learning_rate = 0.01
batch_size = 64
epsi = 0.000000001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

cifar_train_dataset = datasets.CIFAR10(root="./data", download=True, train=True, transform=transform)

cifar_test_dataset = datasets.CIFAR10(root="./data", download=True, train=False, transform=transform)

cifar_train_loader = torch.utils.data.DataLoader(cifar_train_dataset, batch_size=batch_size, shuffle=True)

cifar_test_loader = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=batch_size, shuffle=False)

mnist_train_dataset = datasets.MNIST(root="./data", download=True, train=True, transform=mnist_transform)

mnist_test_dataset = datasets.MNIST(root="./data", download=True, train=False, transform=mnist_transform)

mnist_train_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)

mnist_test_loader = torch.utils.data.DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)


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


def evaluate(test_data, model, pruningMethod):
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
                 'num epochs: ' + str(epoch_number) + '\n' +
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


def training_loop(train_data, test_data, model_name, model, criterion, method):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    if model_name == "lenet5" or model_name == "vgg16":
        old_cost = train_cnn(train_data, model, criterion, optimizer)
    else:
        old_cost = train(train_data, model, criterion, optimizer)
    for _ in tqdm(range(1, epoch_number + 1)):
        if model_name == "lenet5" or model_name == "vgg16":
            new_cost = train_cnn(train_data, model, criterion, optimizer)
        else:
            new_cost = train(train_data, model, criterion, optimizer)
        if abs((new_cost - old_cost)) / new_cost < epsi:
            break
        old_cost = new_cost
    if model_name == "lenet5":
        evaluate_cnn(test_data, model)
        torch.save(model.state_dict(), LENET5_PATH)
    elif model_name == "vgg16":
        evaluate_cnn(test_data, model)
        torch.save(model.state_dict(), VGG16_PATH)
    else:
        evaluate(test_data, model, pruningMethod=method)
        torch.save(model.state_dict(), PATH)


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
                    # print(weight.shape)

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
        if 'weight' in name:
            weights.append(layer)
    # print(weights)
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


def main(arguments):
    print(arguments)
    if arguments.arch == "lenet5":
        model = lenet5.LeNet5(num_classes=10)
        train_data = mnist_train_loader
        test_data = mnist_test_loader
        criterion = nn.CrossEntropyLoss()
    elif arguments.arch == "vgg16":
        model = vgg16.VGG16(num_classes=10)
        train_data = cifar_train_loader
        test_data = cifar_test_loader
        criterion = nn.CrossEntropyLoss()
    else:
        model = classifier.Classifier()
        train_data, test_data = load_data()
        criterion = nn.MSELoss()
    if arguments.train == 'true':
        training_loop(train_data, test_data, model_name=arguments.arch, model=model, criterion=criterion,
                      method=arguments.method)
    if arguments.arch == "lenet5":
        model.load_state_dict(torch.load(LENET5_PATH))
    elif arguments.arch == "vgg16":
        model.load_state_dict(torch.load(VGG16_PATH))
    else:
        model.load_state_dict(torch.load(PATH))
    if arguments.method == 'magnitude':
        for i in range(1, num_iterations + 1):
            prune_rate = (percent ** (1 / i))
            for module in model.named_modules():
                if isinstance(module[1], nn.Linear):
                    prune.l1_unstructured(module[1], 'weight', prune_rate)
                    prune.remove(module[1], 'weight')
            print('Iteration ' + str(i))
            print("Prune rate " + str(prune_rate))
            print('Zero weights ' + str(count_zero_weights(model)))
            training_loop(train_data, test_data, model_name=arguments.arch, model=model, criterion=criterion,
                          method=arguments.method)
            print('Zero weights After training ' + str(count_zero_weights(model)))
    elif arguments.method == 'custom':
        for i in range(1, num_iterations + 1):
            prune_rate = (percent ** (1 / i))
            prune(model, prune_rate)
            print('Iteration ' + str(i))
            print("Prune rate " + str(prune_rate))
            print('Zero weights ' + str(count_zero_weights(model)))
            training_loop(train_data, test_data, model_name=arguments.arch, model=model, criterion=criterion,
                          method=arguments.method)
            print('Zero weights After training ' + str(count_zero_weights(model)))


def count_zero_weights(model):
    zeros = 0
    for param in model.parameters():
        if param is not None:
            zeros += torch.sum((param == 0).int()).data.item()
    return zeros


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='lenet5', choices=['vgg16', 'classifier', 'lenet5'],
                        help='Architecture: vgg16 | classifier | '
                             'lenet5')
    parser.add_argument('--method', default='custom', choices=['custom', 'magnitude'],
                        help='Method used to prune:'
                             'custom | none | '
                             'magnitude')
    parser.add_argument('--train', default='false', choices=['false', 'true'],
                        help='Train model initially before pruning: '
                             'true | false')
    args = parser.parse_args()
    main(arguments=args)
