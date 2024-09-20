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
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

#calculate euclidean distance for four points
def euclidean_distance_4(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2 + (p1[3] - p2[3])**2)
                
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
            
#split dataset into training and testing randomly
def split_data(data):
    shuffled_data = np.random.permutation(data)
    train_data = shuffled_data[0: int(0.8 * len(shuffled_data))]
    test_data = shuffled_data[int(0.8 * len(shuffled_data)):]
    return train_data, test_data

def findNeigbours(train_data, test_data, k):
    distance = []
    for train_ins in train_data:
        dist = euclidean_distance(test_data, train_ins)
        distance.append((train_ins, dist))
    distance.sort(key = lambda x: x[1])
    neighbours = []
    for i in range(k):
        neighbours.append(distance[i][0])
            
    print(neighbours)



#classify class based on neighbours
def knnClassifier(train_data, test_data, k):
    for test_ins in test_data:
        neighbours = findNeigbours(train_data, test_ins, k)
    #neighbours = findNeigbours(train_data, test_data, k)
    #predictData = predict(neighbours)
            
            
#write main function
def main():
    data = read_data('P1input2024.txt')
    data = data.astype(np.int32)
    write_data('P1input2024LongRecords.txt', data)
    long_data = read_data('P1input2024LongRecords.txt')
    long_data = long_data.astype(np.int32)  
    train_set, test_set = split_data(long_data)
    knnClassifier(train_set, test_set, 3)      
    #print(train_set)
    #print(test_set)        
    #print(long_data)
    
if __name__ == '__main__':
    main()