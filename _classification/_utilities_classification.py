#
import os
import numpy as np
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import time


def preprocess(dataframe):
    data_X = dataframe.iloc[0:, :-1]  # Features
    data_y = dataframe.iloc[0:, -1]  # Features
    sc = StandardScaler()
    scaler_train = sc.fit(data_X)
    X_preprocessed = scaler_train.transform(data_X)

    le = LabelEncoder()
    le.fit(data_y)
    y_preprocessed = le.transform(data_y)

    return X_preprocessed, y_preprocessed


def train_mlp(clf, data_train, data_test, label_train, label_test):
    # Preprocess
    train_features, test_features, train_labels, test_labels = data_train, data_test, label_train, label_test

    # Train MLP
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, y_pred)

    return clf, y_pred, test_features, test_labels, accuracy


def save_model(clf, filename):
    # Saving the model
    path_model = 'Save_model/'
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    pickle.dump(clf, open(path_model+filename + '.pkl', 'wb'))


def plot_acc_n_run(accuracies, iteration, filename):
    accuracies = np.array(accuracies)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, iteration+1), accuracies,
             marker='o', linestyle='-', color='b')
    plt.xlabel('Run')
    plt.ylabel('Accuracy')
    plt.title('MLP Classifier Accuracy' + filename +
              ' for ' + str(iteration) + ' Runs')
    plt.grid(True)
    path = 'Save_Figure/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + filename +
                ' MLP Classifier Accuracy for ' + str(iteration) + ' Runs.png')
    plt.show()


class Performance():
    # structure of the solution
    def __init__(self):
        self.accuracies = None
        self.idx_max_acc = None
        self.std_dev = None
        self.mean_acc = None
        self.min_acc = None
        self.max_acc = None


def eval_performance(list_acc, iteration, filename):
    print(list_acc['accuracy'])
    plot_acc_n_run(list_acc['accuracy'], iteration, filename)

    accuracies = np.array(list_acc['accuracy'])
    std_dev = np.std(accuracies)*100
    mean_acc = np.mean(accuracies)*100
    min_acc = np.min(accuracies)*100
    max_acc = np.max(accuracies)
    print('Standard Deviasi: {:.2f}'.format(std_dev))
    print('Mean Accuracies: {:.2f} %'.format(mean_acc))

    idx_max_acc = np.where(list_acc['accuracy'] == max_acc)
    idx_max_acc = np.ravel(np.array(idx_max_acc))
    print('Indices of Max Accuracies: ' + str(idx_max_acc))
    idx = idx_max_acc[-1]  # get last value of idx_max_acc

    # print(list_acc['accuracy'][idx])
    # # print(list_y_pred)
    # print(list_y_pred[idx])
    # print(list_test_features[idx])
    # print(list_test_labels[idx])
    print('Lowest Accuracies: {:.2f} %'.format(min_acc))
    print('Highest Accuracies: {:.2f} %'.format(max_acc*100))

    performance = Performance()
    performance.accuracies = accuracies
    performance.idx_max_acc = idx_max_acc
    performance.std_dev = std_dev
    performance.mean_acc = mean_acc
    performance.min_acc = min_acc
    performance.max_acc = max_acc

    return idx, performance


def evaluation(clf, label_test, predict_test, filename):
    path = 'Save_Figure/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Print AUC
    fpr, tpr, thresholds = roc_curve(label_test, predict_test, pos_label=1)
    plt.figure(1)
    plt.plot(fpr, tpr)
    plt.title("ROC Curve", fontsize=14)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print(f'model AUC score: {roc_auc_score(label_test, predict_test)}')

    plt.savefig(path + filename + ' AUC.png')

    # Print Loss Curve
    plt.figure(2)
    plt.plot(clf.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.savefig(path + filename + ' LC.png')

    # Plot Confusion Matrix
    cm = confusion_matrix(label_test, predict_test, labels=clf.classes_)
    plt.figure(3)
    fig = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=clf.classes_)
    fig.plot()
    plt.savefig(path + filename + ' cm.png')

    # Print Classification Report
    print(classification_report(label_test, predict_test))

    plt.show()


def test_n_classification(clf, data_train, data_test, label_train, label_test, iteration, filename):
    ### SELECT CLASSIFIER ALGORITHM MODEL ###
    ###################################################################################################
    # Architecture Classifier
    clf = clf
    ###################################################################################################

    list_y_pred = []
    list_test_features = []
    list_test_labels = []

    list_acc = {}
    list_acc['accuracy'] = np.zeros(iteration)

    # start timer
    start_time = time.time()
    for i in range(iteration):

        # train classifier
        clf, y_pred, test_features, test_labels, accuracy = train_mlp(
            clf, data_train, data_test, label_train, label_test)

        list_y_pred.append(y_pred)
        list_test_features.append(test_features)
        list_test_labels.append(test_labels)

        list_acc['accuracy'][i] = accuracy
        max_acc = max(list_acc['accuracy'])
        if max_acc >= list_acc['accuracy'][i]:
            save_model(clf, filename)

    idx, performance = eval_performance(list_acc, iteration, filename)

    evaluation(clf, list_test_labels[idx], list_y_pred[idx], filename)
    '''
    ### USE THIS CODE BELOW TO DEBUG OR CHECK PICKLE MODEL ###
    ############################################################################
    # pickled_model = pickle.load(open('Save Model/'+filename+'.pkl', 'rb'))
    # y_predict = pickled_model.predict(list_test_features[idx])

    # # debug
    # if y_predict.all() == list_y_pred[idx].all():
    #      print("oke")
    # else:
    #      print("no")
    #############################################################################
    '''
    # stop timer
    end_time = time.time()
    exec_time = end_time - start_time
    print('Execution time: ' + str(exec_time))

    return performance
