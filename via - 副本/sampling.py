'''Some helper functions
'''
import random
import time
from random import shuffle
random.seed(7)
import numpy as np
from torchvision import datasets, transforms
import codecs
import tensorflow as tf 
import pandas as pd
import os
from datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

def distribute_dataset(label, dataset_name, num_peers, num_classes, dd_type = 'IID', classes_per_peer = 1, samples_per_class = 582,
alpha = 1):
    print("--> Loading of {} dataset".format(dataset_name))
    tokenizer = None
    attack_label = label
    if dataset_name == 'MNIST':
        trainset, testset = get_mnist(dd_type)
    elif dataset_name == 'CIFAR10':
        trainset, testset = get_cifar10()
    elif dataset_name == 'IMDB':
        trainset, testset, tokenizer = get_imdb(num_peers = num_peers)
    elif dataset_name == 'ADULT':
        trainset, testset = get_adult()
    if dd_type == 'IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=1000000)
    elif dd_type == 'NON_IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=alpha)
    elif dd_type == 'EXTREME_NON_IID':
        peers_data_dict, attack_label = sample_extreme(trainset, num_peers, num_classes, classes_per_peer, samples_per_class)

    print("--> Dataset has been loaded!")
    return trainset, testset, peers_data_dict, tokenizer, attack_label
    

# Get the original MNIST data set
def get_mnist(dd_type):
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    transform_noniid = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    curPath = os.path.abspath(os.path.dirname(__file__))

    if dd_type == 'IID':
        trainset = datasets.MNIST('./data', train=True, download=True,
                                  transform=transform)
    else:
        trainset = datasets.ImageFolder(root=curPath + '/dataset_noniid/MNIST', transform=transform_noniid)

    testset = datasets.MNIST('./data', train=False, download=True,
                             transform=transform)
    return trainset, testset

# Get the original CIFAR10 data set
def get_cifar10():
    data_dir = 'data/cifar/'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=transform)

    testset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=transform)
    return trainset, testset

def get_adult():
    # read the adult data set
    data = pd.read_csv('data/adult.csv', sep=',')
    # removing the non values and the two features fnlwgt and education
    data.dropna(inplace=True)
    data = data.drop_duplicates()
    data.reset_index(drop=True, inplace=True)

    # Drop fnlwgt, not interesting for ML
    data.drop('fnlwgt', axis=1, inplace=True)
    data.drop('education', axis=1, inplace=True)
    data.drop('capital-loss', axis=1, inplace=True)
    data.drop('capital-gain', axis=1, inplace=True)
    # incoding the marital-status into married or unmarried
    data['marital-status'].replace('Married-civ-spouse', 'Married', inplace=True)
    data['marital-status'].replace('Divorced', 'Unmarried', inplace=True)
    data['marital-status'].replace('Never-married', 'Unmarried', inplace=True)
    data['marital-status'].replace('Separated', 'Unmarried', inplace=True)
    data['marital-status'].replace('Widowed', 'Unmarried', inplace=True)
    data['marital-status'].replace('Married-spouse-absent', 'Married', inplace=True)
    data['marital-status'].replace('Married-AF-spouse', 'Married', inplace=True)
    obj_columns = data.select_dtypes(['object']).columns
    data[obj_columns] = data[obj_columns].astype('category')

    # Convert numerics to floats 
    num_columns = data.select_dtypes(['int64']).columns
    data[num_columns] = data[num_columns].astype('float64')
    # encoding the categorical attributes into numerical
    marital_status = dict(zip(data['marital-status'].cat.codes, data['marital-status']))
    data['marital-status'] = data['marital-status'].cat.codes
    occupation = dict(zip(data['occupation'].cat.codes, data['occupation']))
    data['occupation'] = data['occupation'].cat.codes
    relationship = dict(zip(data['relationship'].cat.codes, data['relationship']))
    data['relationship'] = data['relationship'].cat.codes
    race = dict(zip(data['race'].cat.codes, data['race']))
    data['race'] = data['race'].cat.codes
    gender = dict(zip(data['gender'].cat.codes, data['gender']))
    data['gender'] = data['gender'].cat.codes
    native_country = dict(zip(data['native-country'].cat.codes, data['native-country']))
    data['native-country'] = data['native-country'].cat.codes
    workclass = dict(zip(data['workclass'].cat.codes, data['workclass']))
    data['workclass'] = data['workclass'].cat.codes

    #changing class from >50K and <=50K to 1 and 0
    data['income'] = data['income'].astype(str)
    data['income'] = data['income'].replace('>50K', 1)
    data['income'] = data['income'].replace('<=50K', 0)

    # convert the data set from pandas to numpy
    data = data.to_numpy()
    # spliting the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(data[:,:-1],data[:,-1], test_size=0.2, random_state=92)
    # normalizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    trainset = AdultDataset(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train))
    testset = AdultDataset(torch.FloatTensor(X_test), 
                       torch.FloatTensor(y_test))
    
    return trainset, testset

#Get the IMDB data set
def get_imdb(num_peers = 10):
    MAX_LEN = 128
    # Read data
    df = pd.read_csv('data/imdb.csv')
    # Convert sentiment columns to numerical values
    df.sentiment = df.sentiment.apply(lambda x: 1 if x=='positive' else 0)
    # Tokenization
    # use tf.keras for tokenization,  
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

        
    train_df = df.iloc[:40000].reset_index(drop=True)
    valid_df = df.iloc[40000:].reset_index(drop=True)

    # STEP 3: pad sequence
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)

    # zero padding
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=MAX_LEN)

    # STEP 4: initialize dataset class for training
    trainset = IMDBDataset(reviews=xtrain, targets=train_df.sentiment.values)

    # initialize dataset class for validation
    testset = IMDBDataset(reviews=xtest, targets=valid_df.sentiment.values)
   
    return trainset, testset, tokenizer


def sample_dirichlet(dataset, num_users, alpha=1):
    classes = {}
    for idx, x in enumerate(dataset):
        _, label = x
        if type(label) == torch.Tensor:
            label = label.item()
        if label in classes:
            classes[label].append(idx)
        else:
            classes[label] = [idx]
    num_classes = len(classes.keys())
    
    peers_data_dict = {i: {'data':np.array([]), 'labels':set()} for i in range(num_users)}

    for n in range(num_classes):
        random.shuffle(classes[n])
        class_size = len(classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(num_users * [alpha]))
        for user in range(num_users):
            num_imgs = int(round(sampled_probabilities[user]))
            sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
            peers_data_dict[user]['data'] = np.concatenate((peers_data_dict[user]['data'], np.array(sampled_list)), axis=0)
            if num_imgs > 0:
                peers_data_dict[user]['labels'].add(n)

            classes[n] = classes[n][min(len(classes[n]), num_imgs):]
   
    return peers_data_dict


def sample_extreme(dataset, num_users, num_classes, classes_per_peer, samples_per_class):
    n = len(dataset)
    num_classes = 10
    peers_data_dict = {i: {'data': np.array([]), 'labels': []} for i in range(num_users)}
    idxs = np.arange(n)
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    label_indices = {l: [] for l in range(num_classes)}
    for l in label_indices:
        label_idxs = np.where(labels == l)
        label_indices[l] = list(idxs[label_idxs])

    labels = [i for i in range(num_classes)]

    for i in range(num_users):
        classes_per_peer = random.randint(4, 6)
        # classes_per_peer = 3
        user_labels = np.random.choice(labels, classes_per_peer, replace=False)
        # classes_per_peer = 3
        # random.seed(time.time())
        # user_labels = random.sample(labels, classes_per_peer)
        for j, l in enumerate(user_labels):
            peers_data_dict[i]['labels'].append(l)
            lab_idxs = label_indices[l][:samples_per_class]

            if j == 1 and i == 0:
                attack_lab_idxs = label_indices[l][:samples_per_class]
                attack_label = l

            label_indices[l] = list(set(label_indices[l]) - set(lab_idxs))
            if len(label_indices[l]) < samples_per_class:
                labels = list(set(labels) - set([l]))
            peers_data_dict[i]['data'] = np.concatenate(
                (peers_data_dict[i]['data'], lab_idxs), axis=0)
            # classes_per_peer = min(len(labels), classes_per_peer)

    for i in range(4):
        if i == 0:
            continue
        if attack_label not in peers_data_dict[i]['labels']:
            peers_data_dict[i]['labels'].append(attack_label)
            peers_data_dict[i]['data'] = np.concatenate(
                (peers_data_dict[i]['data'], attack_lab_idxs), axis=0)

    return peers_data_dict, attack_label
        

