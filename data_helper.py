import torch.utils.data
import torchvision.transforms as transforms
from torchvision import datasets
from dataprocess import load_data

vgg16_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

lenet5_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

alexnet_transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])


def load_data(args):
    if args.arch == "lenet5":
        train_dataset = datasets.MNIST(root="./data", download=True, train=True, transform=lenet5_transform)
        test_dataset = datasets.MNIST(root="./data", download=True, train=False, transform=lenet5_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
        return train_loader, test_loader
    elif args.arch == "vgg16":
        train_dataset = datasets.CIFAR10(root="./data", download=True, train=True, transform=vgg16_transform)
        test_dataset = datasets.CIFAR10(root="./data", download=True, train=False, transform=vgg16_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
        return train_loader, test_loader
    elif args.arch == "alexnet":
        train_dataset = datasets.CIFAR10(root="./data", download=True, train=True, transform=alexnet_transform)
        test_dataset = datasets.CIFAR10(root="./data", download=True, train=False, transform=alexnet_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
        return train_loader, test_loader
    else:
        train_data, test_data = load_data()
        return train_data, test_data
