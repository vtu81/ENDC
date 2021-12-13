from torchvision import datasets

def main():
    data_path = './'
    datasets.MNIST(root=data_path, train=True, download=True)
    datasets.MNIST(root=data_path, train=False, download=True)
    datasets.FashionMNIST(root=data_path, train=True, download=True)
    datasets.FashionMNIST(root=data_path, train=False, download=True)
    datasets.CIFAR10(root=data_path, train=True, download=True)
    datasets.CIFAR10(root=data_path, train=False, download=True)

if __name__ == "__main__":
    main()
