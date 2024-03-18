import argparse
import csv
import joblib
from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def PrintEvalMetrics(predictions, test_indices, y_true):
    num_folds = len(predictions)
    avg_conf_matrix = np.zeros((2, 2))
    avg_accuracy = 0
    avg_precision = 0
    avg_recall = 0

    for i in range(num_folds):
        y_pred = predictions[i]
        y_test = y_true[test_indices[i]]

        # Calculate confusion matrix, accuracy, precision, and recall
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Update running sum of evaluation metrics
        avg_conf_matrix += conf_matrix
        avg_accuracy += accuracy
        avg_precision += precision
        avg_recall += recall

       

    # Calculate averages of evaluation metrics
    avg_conf_matrix /= num_folds
    avg_accuracy /= num_folds
    avg_precision /= num_folds
    avg_recall /= num_folds

    # Print the evaluation metrics
    print("Average Confusion Matrix:\n", avg_conf_matrix)
    print("Average Precision: ", avg_precision)
    print("Average Recall: ", avg_recall)
    print("Average Accuracy: ", avg_accuracy)


# Define a function to perform cross-validation
def CrossFoldValidation( array1_to_pass, array2_to_pass,type="all"):

    X = array1_to_pass
    y = array2_to_pass
    # uncoment this line to print box plot
    # create_individual_boxplot(X)

    clf = svm.SVC()

    pred=[]
    test_indices=[]
    #10-fold cross validation
    kf = KFold(n_splits=10)
    # Loop through each fold
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        #train classifier
        clf.fit(X[train_index], y[train_index])
        #get predictions and save
        pred.append(clf.predict(X[test_index]))
        #save current test index
        test_indices.append(test_index)
    # Save trained model to a file
    # file_name = "file_{}.pkl".format(type)
    # joblib.dump(clf, file_name)
    return pred, test_indices, y

def process_data(original_data, type):
    # Calculate mean, variance, min, and max of each row of the original_data array
    means = np.array([np.mean(x) for x in original_data])
    variances = np.array([np.var(x) for x in original_data])
    mins = np.array([np.min(x) for x in original_data])
    maxs = np.array([np.max(x) for x in original_data])
    # Combine the calculated values into a new array with four columns
    new_data = np.column_stack((means, variances, mins, maxs))
    np.set_printoptions(precision=8, suppress=True,threshold=np.inf)
    if type == "all":
        combined_data = []
        # Concatenate the features at the current index and the next three indices for all data type
        for i in range(0,len(new_data),4):
            combined_features = np.concatenate((new_data[i], new_data[i + 1], new_data[i + 2], new_data[i + 3]))
            combined_data.append(combined_features)
        new_data=np.array(combined_data)

    return new_data

# Process the data based on the given data type
def process_type(array1,target,data,type="all"):
    new_target = []
    if type=="all":
        for i in range(0, len(target), 4):
            chunk = target[i:i+4]
            if all(x == 1 for x in chunk):
                new_target.append(1)
            else:
                new_target.append(0)
        new_target=np.array(new_target)
        return array1,new_target,data
    elif type=="dia":

        indices = []
        # iterate through array1 and find matching elements
        for i, s in enumerate(array1):
            if type == 'dia' and s == 'BP Dia_mmHg':
                indices.append(i)
        
        # use the indices to create new target and d_array
        new_array1 = [array1[i] for i in indices]
        new_target = target[indices]
        new_data = data[indices]
        # print("***********************dia***************************")
        return new_array1,new_target,new_data
    elif type=="sys":
        
        indices = []
        # iterate through array1 and find matching elements
        for i, s in enumerate(array1):
            if type == 'sys' and s == 'LA Systolic BP_mmHg':
                indices.append(i)

        
        # use the indices to create new target and d_array
        new_array1 = [array1[i] for i in indices]
        new_target = target[indices]
        new_data = data[indices]
        # print("***********************sys***************************")
        return new_array1,new_target,new_data
    elif type=="eda":

        indices = []
        # iterate through array1 and find matching elements
        for i, s in enumerate(array1):
            if type == 'eda' and s == 'EDA_microsiemens':
                indices.append(i)

        
        # use the indices to create new target and d_array
        new_array1 = [array1[i] for i in indices]
        new_target = target[indices]
        new_data = data[indices]
        # print("***********************eda***************************")
        return new_array1,new_target,new_data
    elif type=="res":
        
        indices = []
        # iterate through array1 and find matching elements
        for i, s in enumerate(array1):
            if type == 'res' and s == 'Respiration Rate_BPM':
                indices.append(i)
        
        # use the indices to create new target and d_array
        new_array1 = [array1[i] for i in indices]
        new_target = target[indices]
        new_data = data[indices]
        # print("***********************res***************************")
        return new_array1,new_target,new_data
    else:
        print("***********************Data type given wrong***********************")

    
def create_boxplot(data):
    # extract the data for each sub-element
    means = data[:, 0]
    variances = data[:, 1]
    mins = data[:, 2]
    maxs = data[:, 3]

    # create a list of the data groups
    groups = [means, variances, mins, maxs]

    # create the box plot
    fig, ax = plt.subplots()
    ax.boxplot(groups)

    # add labels to the plot
    ax.set_xticklabels(['Mean', 'Variance', 'Min', 'Max'])
    ax.set_ylabel('Value')

    # display the plot
    plt.show()





# Read data from csv and save it in multiple arrays
def process_csv(type,filename):
    with open(filename,'r') as f:
        reader = csv.reader(f)
        array0 = []
        array1 = []
        array2 = []
        array3 = []
        for row in reader:
            array0.append(row[0])
            array1.append(row[1])
            array2.append(row[2])
            array3.append([float(x) for x in row[3:]])
        # Define a mapping dictionary to convert the string labels in array2 to integers
        mapping = {
            'Pain': 0,
            'No Pain': 1,
        }
        # Convert the labels in array2 to integers using the mapping dictionary
        target = np.array([mapping[value] for value in array2])

        # Uncoment this line to print line graph
        # plot_rows(array3)
        # plot_rows_individual(array3)

        # Process the data in array3 to calculate the features (mean, variance, min, max)
        data=process_data(array3,type)
        # Process the data in array1, target, and data based on the input type
        array1,target,data=process_type(array1,target,data,type)
    return array0, array1, target, array3,data


def create_individual_boxplot(data):
    # extract the data for each sub-element
    means = data[:, 0]
    variances = data[:, 1]
    mins = data[:, 2]
    maxs = data[:, 3]

    # create a list of the data groups
    groups = [means, variances, mins, maxs]
    labels = ['Mean', 'Variance', 'Min', 'Max']

    # create a single figure with four subplots
    fig, axs = plt.subplots(1, 4, figsize=(15, 6))

    # create a box plot for each label in a separate subplot
    for i in range(len(groups)):
        axs[i].boxplot(groups[i])
        axs[i].set_title(labels[i])
        # axs[i].set_ylabel(labels[i])
    plt.subplots_adjust(wspace=0.4)
    # display the plot
    plt.show()




def plot_rows(array):
    # create a single subplot
    fig, ax = plt.subplots(figsize=(7, 7))

    # define colors for each line
    colors = ['blue', 'orange', 'green', 'red']

    # define titles for each line
    titles = ['DIA', 'EDA', 'SYS', 'RES']

    # loop through each row and plot the corresponding line with a different color for each row
    for i in range(4):
        row = np.array(array[i+68])
        print(row[1])
        ax.plot(row, color=colors[i], label=titles[i])

    # set the title and labels
    ax.set_title("All Lines")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")

    # add legend and adjust spacing
    ax.legend()
    fig.tight_layout()

    # display the plot
    plt.show()


def plot_rows_individual(array):
    # create four subplots
    fig, axes = plt.subplots(4, 1, figsize=(7, 7))

    # define colors for each line
    colors = ['blue', 'orange', 'green', 'red']

    # define titles for each subplot
    titles = ['DIA', 'EDA', 'SYS', 'RES']

    # loop through each subplot and plot the corresponding row with a different color for each line
    for i, ax in enumerate(axes):
        row = np.array(array[i+68])
        ax.plot(row, color=colors[i])
        ax.set_title(titles[i])
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")

    # adjust spacing between subplots
    fig.tight_layout()

    # display the plot
    plt.show()

def main(type,directoryName):
    # Set NumPy printing options to show up to 8 decimal places and suppress scientific notation
    np.set_printoptions(precision=8, suppress=True,threshold=np.inf)
    filename = directoryName
    # Process the CSV file and store the results in arrays
    array0, array1, array2, array3,data = process_csv(type,filename)
    # Perform cross-fold validation on the processed data and store the predictions, test indices, and actual labels
    pred, test_indices, y = CrossFoldValidation(data,array2,type)
    # Call to PrintEvalMetrics function to print evaluation metrics based on the predictions, test indices, and actual labels
    PrintEvalMetrics(pred, test_indices, y)

if __name__ == '__main__':
    # Create an argument parser object with a description for the program
    parser = argparse.ArgumentParser(description='Project 2')
    # command-line arguments to the parser: data type and directoryName
    parser.add_argument('type', nargs='?', type=str, default='Original', help='Data Type: If not given Original is default')
    parser.add_argument('directoryName', type=str, default='data.csv', help='Directory Name: If not given BU4DFE_BND_V1.1 is default')
    args = parser.parse_args()
    main(args.type,args.directoryName)