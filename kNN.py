import numpy as np
from scipy.io import loadmat


def kNN(training_images, training_labels, testing_images, testing_labels, k, return_inaccurate_indexes = False):
    class_accuracies = [0] * 10
    accuracy = 0
    inaccurate_index = []
    for index, image in enumerate(testing_images):
        nearest_neighbor_indices = find_nearest_neighbors(image, training_images, k)
        label = take_neighborhood_vote(training_labels[nearest_neighbor_indices])
        if label == testing_labels[index]:
            accuracy += 1
            class_accuracies[label] += 1
        else:
            inaccurate_index.append(index)
    for index in range(10):
        class_accuracies[index] = class_accuracies[index] / (testing_labels == [index]).sum()
    print(type(testing_labels))
    accuracy = round(accuracy / testing_labels.size, 2)
    if return_inaccurate_indexes:
        return np.round_(class_accuracies, decimals=2), accuracy, inaccurate_index
    return np.round_(class_accuracies, decimals=2), accuracy


def find_nearest_neighbors(target_image, training_images, k):
    """ returns indices of the k nearest neighbors"""
    distances = []  # list of tuples in the format: (index of neighbor, distance)
    for index, image in enumerate(training_images):
        distance = np.linalg.norm(target_image - image, ord=2) # Euclidean distance
        distances.append((index, distance))
    distances.sort(key = lambda x: x[1])  # sort list by second element of the tuples
    distances = np.array([x[0] for x in distances])  # extract only the indexes of the neighbors
    return distances[:k]


def take_neighborhood_vote(nearest_neighbor_labels):
    """ returns most common label of nearest neighbors"""
    labels, counts = np.unique(nearest_neighbor_labels, return_counts=True)
    inds = np.random.permutation(len(counts))  # randomly permute list so that 
    counts = counts[inds]                      # if two labels are tied, one is chosen randomly
    labels = labels[inds]
    top_vote = np.argmax(counts)
    return labels[top_vote]


if __name__ == "__main__":
    #Loading the data
    M = loadmat('MNIST_digit_data.mat')
    images_train,images_test,labels_train,labels_test= M['images_train'], \
        M['images_test'],M['labels_train'],M['labels_test']
    #just to make all random sequences on all computers the same.
    np.random.seed(1)
    #randomly permute data points
    inds = np.random.permutation(images_train.shape[0])
    images_train = images_train[inds]
    labels_train = labels_train[inds]


    inds = np.random.permutation(images_test.shape[0])
    images_test = images_test[inds]
    labels_test = labels_test[inds]
    print(kNN(images_train[:100], labels_train[:100], images_test[:100], labels_test[:100], 3))
