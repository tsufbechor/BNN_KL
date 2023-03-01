import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy
import pickle
from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator
from scipy.stats import bernoulli

train_dataset = dsets.MNIST(root="./data",
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True
                            )
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

test_dataset = dsets.MNIST(root="./data",
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True
                           )
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          shuffle=True)


@variational_estimator
class BayesianCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5, 5))
        self.conv2 = BayesianConv2d(6, 16, (5, 5))
        self.fc1 = BayesianLinear(256, 120)
        self.fc2 = BayesianLinear(120, 84)
        self.fc3 = BayesianLinear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def full_model():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier = BayesianCNN().to(device)
        num_of_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        kl_fullmodel=[]
        train_accuracy = []
        test_accuracy = []
        correct_train = 0
        total_train = 0
        correct_test=0
        total_test = 0
        for epoch in range(10):
            for i, (datapoints, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = classifier.sample_elbo(inputs=datapoints.to(device),
                                              labels=labels.to(device),
                                              criterion=criterion,
                                              sample_nbr=3,
                                              complexity_cost_weight=1 / 50000)
                loss.backward()
                optimizer.step()
                outputs = classifier(datapoints.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels.to(device)).sum().item()
            train_accuracy.append(correct_train / total_train)
            for data in test_loader:
                images, labels = data
                outputs = classifier(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels.to(device)).sum().item()
            test_accuracy.append(correct_test / total_test)
            kl_fullmodel.append(kl_divergence_from_nn(classifier).item())
        with open("q2_full_model.pkl", "wb") as f:
            pickle.dump(classifier, f)
        # Plotting the KL divergence
        plt.plot(kl_fullmodel)
        plt.xlabel('Epochs')
        plt.ylabel('KL divergence')
        plt.title('KL divergence vs Epochs - All Samples')
        plt.show()
        kl_divided_by_number_of_parameters = [x / num_of_parameters for x in kl_fullmodel]
        # Plotting the KL divergence divided by the number of parameters
        plt.plot(kl_divided_by_number_of_parameters)
        plt.xlabel('Epochs')
        plt.ylabel('KL divergence divided by the number of parameters')
        plt.title('KL divergence average vs Epochs- All Samples')
        plt.show()
        # Plotting the train and test accuracy
        plt.plot(train_accuracy, label='Train Accuracy')
        plt.plot(test_accuracy, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs- All Samples')
        plt.legend()
        plt.show()
        print("Full Model")
        print("Mean of KL divided by parameter is: ", sum(kl_divided_by_number_of_parameters) / len(kl_divided_by_number_of_parameters))
        print("Mean of KL is: ", sum(kl_fullmodel) / len(kl_fullmodel))
        print("Mean of train accuracy is: ", sum(train_accuracy) / len(train_accuracy))
        print("Mean of test accuracy is: ", sum(test_accuracy) / len(test_accuracy))
def first_200():
    train_dataset = dsets.MNIST(root="./data",
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True
                                )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=200,
                                               shuffle=False)

    test_dataset = dsets.MNIST(root="./data",
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True
                               )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=200,
                                              shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = BayesianCNN().to(device)
    num_of_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    kl_first200 = []
    train_accuracy = []
    test_accuracy = []
    correct_train = 0
    total_train = 0
    correct_test = 0
    total_test = 0
    datapoints, labels = next(iter(train_loader))
    for epoch in range(100):
        optimizer.zero_grad()
        loss = classifier.sample_elbo(inputs=datapoints.to(device),
                                      labels=labels.to(device),
                                      criterion=criterion,
                                      sample_nbr=3,
                                      complexity_cost_weight=1 / 50000)
        loss.backward()
        optimizer.step()
        outputs = classifier(datapoints.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels.to(device)).sum().item()
        train_accuracy.append(correct_train / total_train)
        with torch.no_grad():
            for data in test_loader:
                images2, labels2 = data
                outputs = classifier(images2.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels2.size(0)
                correct_test += (predicted == labels2.to(device)).sum().item()
        test_accuracy.append(correct_test / total_test)
        kl_first200.append(kl_divergence_from_nn(classifier).item())
    with open("q2_first_200.pkl", "wb") as f:
        pickle.dump(classifier, f)
    # Plotting the KL divergence
    plt.plot(kl_first200)
    plt.xlabel('Epochs')
    plt.ylabel('KL divergence')
    plt.title('KL divergence vs Epochs- First 200 samples')
    plt.show()
    # Plotting the train and test accuracy
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs-First 200 samples')
    plt.legend()
    plt.show()
    # Plotting the KL divergence divided by the number of parameters
    kl_divided_by_number_of_parameters = [x / num_of_parameters for x in kl_first200]
    plt.plot(kl_divided_by_number_of_parameters)
    plt.xlabel('Epochs')
    plt.ylabel('KL divergence divided by the number of parameters')
    plt.title('KL divergence average vs Epochs- First 200 samples')
    plt.show()
    print("First 200 Samples")
    print("Mean of KL divided by parameter is: ", sum(kl_divided_by_number_of_parameters) / len(kl_divided_by_number_of_parameters))
    print("Mean of KL is: ", sum(kl_first200) / len(kl_first200))
    print("Mean of train accuracy is: ", sum(train_accuracy) / len(train_accuracy))
    print("Mean of test accuracy is: ", sum(test_accuracy) / len(test_accuracy))
def first_200_only_2_8():
        train_dataset = dsets.MNIST(root="./data",
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True
                                    )
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=200,
                                                   shuffle=False)

        test_dataset = dsets.MNIST(root="./data",
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   download=True
                                   )
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=200,
                                                  shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classifier = BayesianCNN().to(device)
        num_of_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        kl_first200 = []
        datapoints = torch.zeros(200, 1, 28, 28)
        labels = torch.zeros(200)
        train_accuracy = []
        test_accuracy = []
        test_only_2_8 = []
        correct_train = 0
        total_train = 0
        correct_test = 0
        total_test = 0
        correct_test_2 = 0
        total_test_2 = 0
        test_points, test_labels = next(iter(test_loader))
        counter=0
        for i, (datapoints1, labels1) in enumerate(train_loader):
            for j, label in enumerate(labels1):
                if (label.item() == 3 or label.item() == 8) and counter<200:
                    datapoints[counter] = datapoints1[j]
                    labels[counter] = label
                    counter += 1
                if counter == 200:
                        break
        counter=0
        labels=labels.long()
        for i, (temp_points, temp_labels) in enumerate(test_loader):
            for j, label in enumerate(temp_labels):
                    if (label.item() == 3 or label.item() == 8) and counter < 200:
                        test_points[counter] = temp_points[j]
                        test_labels[counter] = label
                        counter += 1
                    if counter == 200:
                        break
        test_labels=test_labels.long()
        for epoch in range(100):
            optimizer.zero_grad()
            loss = classifier.sample_elbo(inputs=datapoints.to(device),
                                          labels=labels.to(device),
                                          criterion=criterion,
                                          sample_nbr=3,
                                          complexity_cost_weight=1 / 50000)
            loss.backward()
            optimizer.step()
            outputs = classifier(datapoints.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels.to(device)).sum().item()
            train_accuracy.append(correct_train / total_train)
            with torch.no_grad():
                for data in test_loader:
                    #Test on all data set
                    images2, labels2 = data
                    outputs = classifier(images2.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels2.size(0)
                    correct_test += (predicted == labels2.to(device)).sum().item()
                   #Test for only 2,8 digits
                    outputs_test = classifier(test_points.to(device))
                    _, predicted_test = torch.max(outputs_test.data, 1)
                    total_test_2 += test_labels.size(0)
                    correct_test_2 += (predicted_test == test_labels.to(device)).sum().item()

            test_accuracy.append(correct_test / total_test)
            test_only_2_8.append(correct_test_2 / total_test_2)
            kl_first200.append(kl_divergence_from_nn(classifier).item())
        with open("q2_first_200_only_2_8.pkl", "wb") as f:
            pickle.dump(classifier, f)
        # Plotting the KL divergence
        plt.plot(kl_first200)
        plt.xlabel('Epochs')
        plt.ylabel('KL divergence')
        plt.title('KL divergence vs Epochs- First 200 samples of 2 and 8')
        plt.show()
        # Plotting the train and test accuracy
        plt.plot(train_accuracy, label='Train Accuracy')
        plt.plot(test_only_2_8, label='Test Accuracy for only 2 and 8')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs for only first 200 samples of 2 and 8')
        plt.legend()
        plt.show()
        # Plotting the KL divergence divided by the number of parameters
        kl_divided_by_number_of_parameters = [x / num_of_parameters for x in kl_first200]
        plt.plot(kl_divided_by_number_of_parameters)
        plt.xlabel('Epochs')
        plt.ylabel('KL divergence divided by the number of parameters')
        plt.title('KL divergence average vs Epochs- First 200 samples of 2 and 8')
        plt.show()
        print("First 200 samples of 2 and 8")
        print("Mean of KL divided by number of parameters is: ", sum(kl_divided_by_number_of_parameters) / len(kl_divided_by_number_of_parameters))
        print("Mean of KL is: ", sum(kl_first200) / len(kl_first200))
        print("Mean of train accuracy is: ", sum(train_accuracy) / len(train_accuracy))
        print("Mean of test accuracy is: ", sum(test_only_2_8) / len(test_only_2_8))


def only_2_8():
    train_dataset = dsets.MNIST(root="./data",
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True
                                )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=200,
                                               shuffle=False)

    test_dataset = dsets.MNIST(root="./data",
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True
                               )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=200,
                                              shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx_train = (train_dataset.targets == 8) | (train_dataset.targets == 3)
    train_dataset.data = train_dataset.data[idx_train]
    train_dataset.targets = train_dataset.targets[idx_train]
    idx_test = (test_dataset.targets == 8) | (test_dataset.targets == 3)
    test_dataset.data = test_dataset.data[idx_test]
    test_dataset.targets = test_dataset.targets[idx_test]
    classifier = BayesianCNN().to(device)
    num_of_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    kl_only_2_8 = []
    train_accuracy = []
    test_accuracy = []
    test_only_2_8 = []
    correct_train = 0
    total_train = 0
    correct_test = 0
    total_test = 0
    for epoch in range(10):
        for i, (datapoints, labels) in enumerate(train_loader):
             optimizer.zero_grad()
             loss = classifier.sample_elbo(inputs=datapoints.to(device),
                                                      labels=labels.to(device),
                                                      criterion=criterion,
                                                      sample_nbr=3,
                                                      complexity_cost_weight=1 / 50000)
             loss.backward()
             optimizer.step()
             outputs = classifier(datapoints.to(device))
             _, predicted = torch.max(outputs.data, 1)
             total_train += labels.size(0)
             correct_train += (predicted == labels.to(device)).sum().item()
        train_accuracy.append(correct_train / total_train)
        with torch.no_grad():
            for data in test_loader:
                # Test on all data set
                images, labels = data
                outputs = classifier(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels.to(device)).sum().item()
        test_accuracy.append(correct_test / total_test)
        kl_only_2_8.append(kl_divergence_from_nn(classifier).item())
    with open("q2_only_2_8.pkl", "wb") as f:
        pickle.dump(classifier, f)
    # Plotting the KL divergence
    plt.plot(kl_only_2_8)
    plt.xlabel('Epochs')
    plt.ylabel('KL divergence')
    plt.title('KL divergence vs Epochs- All samples of 2 and 8')
    plt.show()
    # Plotting the train and test accuracy
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy for all numbers')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs for all samples 2 and 8')
    plt.legend()
    plt.show()
    kl_divided_by_number_of_parameters = [x / num_of_parameters for x in kl_only_2_8]
    plt.plot(kl_divided_by_number_of_parameters)
    plt.xlabel('Epochs')
    plt.ylabel('KL divergence averaged by number of parameters')
    plt.title('KL divergence average vs Epochs- All samples of 2 and 8')
    plt.show()
    print("All samples of 2 and 8")
    print("Mean of KL divided by number of parameters is: ", sum(kl_divided_by_number_of_parameters) / len(kl_divided_by_number_of_parameters))
    print("Mean of KL is: ", sum(kl_only_2_8) / len(kl_only_2_8))
    print("Mean of train accuracy is: ", sum(train_accuracy) / len(train_accuracy))
    print("Mean of test accuracy is: ", sum(test_accuracy) / len(test_accuracy))
def random_labels():
    train_dataset = dsets.MNIST(root="./data",
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True
                                )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=200,
                                               shuffle=False)

    test_dataset = dsets.MNIST(root="./data",
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True
                               )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=200,
                                                shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_labels_train = bernoulli.rvs(0.5, size=200)
    random_labels_train_tensor = torch.Tensor(random_labels_train).long()
    random_labels_test = torch.Tensor(bernoulli.rvs(0.5, size=len(test_dataset.targets))).long()
    classifier = BayesianCNN().to(device)
    number_of_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    kl_random = []
    train_accuracy = []
    test_accuracy = []
    correct_train = 0
    total_train = 0
    correct_test = 0
    total_test = 0
    datapoints, labels = next(iter(train_loader))
    for epoch in range(100):
            optimizer.zero_grad()
            loss = classifier.sample_elbo(inputs=datapoints.to(device),
                                          labels=random_labels_train_tensor.to(device),
                                          criterion=criterion,
                                          sample_nbr=3,
                                          complexity_cost_weight=1 / 50000)
            loss.backward()
            optimizer.step()
            outputs = classifier(datapoints.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total_train += random_labels_train_tensor.size(0)
            correct_train += (predicted == random_labels_train_tensor.to(device)).sum().item()
            train_accuracy.append(correct_train / total_train)
            with torch.no_grad():
                for data in test_loader:
                    images2, labels2 = data
                    outputs = classifier(images2.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels2.size(0)
                    correct_test += (predicted == random_labels_train_tensor.to(device)).sum().item()
                test_accuracy.append(correct_test / total_test)
            kl_random.append(kl_divergence_from_nn(classifier).item())

    with open("q2_random.pkl", "wb") as f:
        pickle.dump(classifier, f)
# Plotting the KL divergence
    plt.plot(kl_random)
    plt.xlabel('Epochs')
    plt.ylabel('KL divergence')
    plt.title('KL divergence vs Epochs- Random labels')
    plt.show()
    kl_divided_by_number_of_parameters = [x / number_of_parameters for x in kl_random]
    plt.plot(kl_divided_by_number_of_parameters)
    plt.xlabel('Epochs')
    plt.ylabel('KL divergence averaged by number of parameters')
    plt.title('KL divergence average vs Epochs- Random labels')
    plt.show()
    # Plotting the train and test accuracy
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs-Random')
    plt.legend()
    plt.show()
    print("Random labels")
    print("Mean of KL divided by number of parameters is: ", sum(kl_divided_by_number_of_parameters) / len(kl_divided_by_number_of_parameters))
    print("Mean of KL is: ", sum(kl_random) / len(kl_random))
    print("Mean of train accuracy is: ", sum(train_accuracy) / len(train_accuracy))
    print("Mean of test accuracy is: ", sum(test_accuracy) / len(test_accuracy))


full_model()
first_200()
only_2_8()
first_200_only_2_8()
random_labels()

