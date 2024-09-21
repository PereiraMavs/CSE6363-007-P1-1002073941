import numpy as np
import math

#list of all the tree locations
trees = [[4, 0], [0, 4], [3, 9], [7, 9]]

#read data
def read_data(file_name):
    with open(file_name) as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    data = [x.split() for x in data]
    data = np.array(data)
    return data

#calculate manhattan distance
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

#calculate euclidean distance for two points
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

#calculate euclidean distance for three points
def euclidean_distance_3(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[3] - p2[3])**2)

#calculate euclidean distance for four points
def euclidean_distance_4(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[3] - p2[3])**2 + (p1[4] - p2[4])**2)
                
#find out minimum manhattan distance
def find_min_distance_tree(data):
    min_distance = math.inf
    distance = 0
    min_index = 0
    for j in range(len(trees)):
        distance = manhattan_distance(data, trees[j])
        if distance < min_distance:
            min_distance = distance
            min_index = j + 1
    return [min_distance, min_index]
                
#write manhattan distance in a new file
def write_data(file_name, data):
    with open(file_name, 'w') as f:
        for i in range(len(data)):
            tree_data = find_min_distance_tree(data[i])
            f.write(str(data[i][0]) + ' ' + str(data[i][1]) + ' ' 
                    + str(data[i][2]) + ' ' + str(tree_data[0]) + ' ' + str(tree_data[1]) + '\n')
#append data to file
def append_data(file_name, data):
    with open(file_name, 'a') as f:
        f.write(str(data) + '\n')


#split dataset into training and testing randomly
def split_data(data):
    shuffled_data = np.random.permutation(data)
    train_data = shuffled_data[0: int(0.8 * len(shuffled_data))]
    test_data = shuffled_data[int(0.8 * len(shuffled_data)):]
    return train_data, test_data

def findNeigbours(train_data, test_data, k, input_vector = 2):
    distance = []
    for train_ins in train_data:
        if input_vector == 2:
            dist = euclidean_distance(test_data, train_ins)
        elif input_vector == 3:
            dist = euclidean_distance_3(test_data, train_ins)
        elif input_vector == 4:
            dist = euclidean_distance_4(test_data, train_ins)
        distance.append((train_ins, dist))
    #sort the distance array based on second element
    distance.sort(key = lambda x: x[1])
    neighbours = []
    for i in range(k):
        neighbours.append(distance[i][0])
        #print(distance[i][1])
            
    #print(neighbours)
    return neighbours

#predict class based on neighbours
def predict(neighbours):
    classes = {}
    for i in range(len(neighbours)):
        response = neighbours[i][2]
        if response in classes:
            classes[response] += 1
        else:
            classes[response] = 1
    sortedVotes = sorted(classes.items(), key = lambda x: x[1], reverse = True)
    #print(sortedVotes)
    return sortedVotes[0][0]

#classify class based on neighbours
def knnClassifier(train_data, test_data, k, input_vector = 2):
    predictData = []
    for test_ins in test_data:
        neighbours = findNeigbours(train_data, test_ins, k, input_vector)
        pre = predict(neighbours)
        predictData.append(pre)
    #print(predictData)
    return predictData
            
#calculate accuracy
def accuracy(test_data, predictData):
    correct = 0
    for i in range(len(test_data)):
        if test_data[i][2] == predictData[i]:
            correct += 1
    return (correct/float(len(test_data))) * 100.0

#calculate recall
def recall(test_data, predictData):
    correct = 0
    true_positive = 0
    false_negative = 0
    for i in range(len(test_data)):
        if test_data[i][2] == predictData[i]:
            correct += 1
            if test_data[i][2] == 1:
                true_positive += 1
        elif predictData[i] == 0:
            false_negative += 1
    return (true_positive/float(true_positive + false_negative)) * 100.0

#calculate precision
def precision(test_data, predictData):
    correct = 0
    true_positive = 0
    false_positive = 0
    for i in range(len(test_data)):
        if test_data[i][2] == predictData[i]:
            correct += 1
            if test_data[i][2] == 1:
                true_positive += 1
        elif predictData[i] == 1:
            false_positive += 1
    return (true_positive/float(true_positive + false_positive)) * 100.0

#calculate f1 score
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

#calcualte confusion matrix
def confusion_matrix(test_data, predictData):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(test_data)):
        if test_data[i][2] == predictData[i]:
            if test_data[i][2] == 1:
                true_positive += 1
            else:
                true_negative += 1
        elif predictData[i] == 1:
            false_positive += 1
        else:
            false_negative += 1
    return [[true_negative, false_positive], [false_negative, true_positive]]

#calculate all metrics
def calculate_metrics(test_data, train_set, test_set, k, input_encoding):
    append_data('P1Output2024.txt', 'Question 2 ' + 'K = ' + str(k) + ' input vector set ' + str(input_encoding))
    result = knnClassifier(train_set, test_set, k, input_encoding)
    acc = accuracy(test_data, result)
    precision_val = precision(test_data, result)
    recall_val = recall(test_data, result)
    f1_score_val = f1_score(precision_val, recall_val)
    confusion_matrix_val = confusion_matrix(test_data, result)
    append_data('P1Output2024.txt', 'Confusion Matrix: ' + str(confusion_matrix_val))
    append_data('P1Output2024.txt', 'Accuracy: ' + str(acc))
    append_data('P1Output2024.txt', 'Precision: ' + str(precision_val))
    append_data('P1Output2024.txt', 'Recall: ' + str(recall_val))
    append_data('P1Output2024.txt', 'F1 Score: ' + str(f1_score_val))
    append_data('P1Output2024.txt', '\n')
    print('Question 2 ' + 'K = ' + str(k) + ' input vector set ' + str(input_encoding))
    print('Confusion Matrix: ', confusion_matrix_val)
    print('Accuracy: ', acc)
    print('Precision: ', precision_val)
    print('Recall: ', recall_val)
    print('F1 Score: ', f1_score_val)
    print('\n')


def main():
    data = read_data('P1input2024.txt')
    data = data.astype(np.int32)
    write_data('P1input2024LongRecords.txt', data)
    long_data = read_data('P1input2024LongRecords.txt')
    long_data = long_data.astype(np.int32)  
    train_set, test_set = split_data(long_data)

    calculate_metrics(test_set, train_set, test_set, 3, 2)
    calculate_metrics(test_set, train_set, test_set, 5, 2)
    calculate_metrics(test_set, train_set, test_set, 7, 2)

    calculate_metrics(test_set, train_set, test_set, 3, 3)
    calculate_metrics(test_set, train_set, test_set, 5, 3)
    calculate_metrics(test_set, train_set, test_set, 7, 3)

    calculate_metrics(test_set, train_set, test_set, 3, 4)
    calculate_metrics(test_set, train_set, test_set, 5, 4)
    calculate_metrics(test_set, train_set, test_set, 7, 4)
    
if __name__ == '__main__':
    main()