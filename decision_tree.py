# Spring 2022 10601 HW2
# Author: Xin Zheng

import numpy as np
import sys
import math


class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self, left,right,attr,vote, attr_val, depth, type, label1, label1_count, label2, label2_count):
        self.left = left
        self.right = right
        self.attr = attr
        self.attr_val = attr_val
        self.vote = vote
        self.depth = depth
        self.type = type
        self.label1 = label1
        self.label1_count = label1_count
        self.label2 = label2
        self.label2_count = label2_count

def read_in_file(filepath):
    '''
    This function read in tab-separated files and return the data as nparray
    :param filepath: the path to the file
    :return: a nparray storing data in the file
    '''
    file = np.genfromtxt(filepath,dtype=str,delimiter='\t')
    return file

def count_label(data):
    '''
    This function count the number of each label
    :param data: np array of politician data
    :return: the number of label1, the number of label2
    '''
    label1 = data[1][-1]
    label2 = ''
    label1_count = 0
    label2_count = 0
    for i in range(1, len(data)):
        row = data[i]
        if row[-1] == label1:
            label1_count += 1
        else:
            if label2 == '':
                label2 = row[-1]
            label2_count += 1
    return label1, label1_count, label2, label2_count

def majority_vote(label1, label1_count, label2, label2_count):
    '''
    find the most common label
    :param data: np array of data
    :return: the label that appears the most times in the data
    '''
    if label1_count > label2_count:
        return label1
    elif label1_count < label2_count:
        return label2
    else:
        return label2 if label1 < label2 else label1

def get_entropy(label1_count,label2_count):
    '''
    Calculate the entropy before splitting
    :param label1: first label
    :param label1_count: number of label1
    :param label2: second label
    :param label2_count: number of label2
    :return: the entropy of the set
    '''
    if label1_count == 0 or label2_count ==0:
        return 0
    total_num = label1_count + label2_count
    return -(label1_count/total_num)*math.log2(label1_count/total_num)-(label2_count/total_num)*math.log2(label2_count/total_num)

def count_attr_val(data,attr,label1,label2):
    val1_label1_count = 0
    val1_label2_count = 0
    val2_label1_count = 0
    val2_label2_count = 0
    attr_val1 = data[1][attr]
    attr_val2 = ''
    val1_count = 0
    val2_count = 0
    for i in range(1, len(data)):
        row = data[i]
        if row[attr] == attr_val1:
            val1_count += 1
            if row[-1] == label1:
                val1_label1_count += 1
            if row[-1] == label2:
                val1_label2_count += 1
        else:
            if attr_val2 == '':
                attr_val2 = row[attr]
            val2_count += 1
            if row[-1] == label1:
                val2_label1_count += 1
            if row[-1] == label2:
                val2_label2_count += 1
    return attr_val1, val1_count, val1_label1_count, val1_label2_count, attr_val2,val2_count, val2_label1_count, val2_label2_count


def get_mutual_info(val1_count, val2_count, val1_label1_count, val1_label2_count, val2_label1_count, val2_label2_count, entropy):
    val1_entropy = get_entropy(val1_label1_count, val1_label2_count)
    val2_entropy = get_entropy(val2_label1_count, val2_label2_count)

    val1_prob = val1_count / (val1_count + val2_count)
    val2_prob = val2_count / (val1_count + val2_count)

    attr1_mutual_info = entropy - val1_entropy * val1_prob - val2_entropy * val2_prob
    return(attr1_mutual_info)

def get_mutual_info_all_attr(data,label1,label2,entropy):
    result = {}
    for i in range(0,len(data[0])-1):
        attr_val1, val1_count, val1_label1_count, val1_label2_count, attr_val2, val2_count, val2_label1_count, val2_label2_count = count_attr_val(
            data, i, label1, label2)
        attr_mutual_info = get_mutual_info(val1_count, val2_count, val1_label1_count, val1_label2_count,
                                            val2_label1_count, val2_label2_count, entropy)
        result[data[0][i]] = [i, attr_val1, val1_count,attr_val2, val2_count,attr_mutual_info]
    return result

def get_attr_to_split_on(result):
    max = -sys.maxsize - 1
    max_attr = ''
    max_idx = 0
    for attr in result:
        if result[attr][-1] > max:
            max = result[attr][-1]
            max_attr = attr
            max_idx = result[attr][0]
    return max_attr, max_idx

def train_decision_tree(train_data,max_depth):
    label1, label1_count, label2, label2_count = count_label(train_data)
    entropy = get_entropy(label1_count, label2_count)
    vote = majority_vote(label1, label1_count, label2, label2_count)

    result = get_mutual_info_all_attr(train_data, label1, label2, entropy)
    max_attr, max_idx = get_attr_to_split_on(result)

    root = Node(None,None,max_attr,vote, None, 0, 'root', label1, label1_count, label2, label2_count)
    left_data = np.delete(train_data, np.where(train_data[:,max_idx] == result[max_attr][3])[0],0)
    right_data = np.delete(train_data,np.where(train_data[:,max_idx] == result[max_attr][1])[0],0)
    left_data = np.delete(left_data, max_idx, 1)
    right_data = np.delete(right_data,max_idx,1)

    left = train_decision_tree_helper(left_data, result[max_attr][1], root.depth+1,max_depth)

    right = train_decision_tree_helper(right_data,result[max_attr][3], root.depth+1,max_depth)
    root.left = left
    root.right = right
    return root

def train_decision_tree_helper(data, attr_val, depth,max_depth):
    label1, label1_count, label2, label2_count = count_label(data)
    entropy = get_entropy(label1_count, label2_count)
    vote = majority_vote(label1, label1_count, label2, label2_count)
    result = get_mutual_info_all_attr(data, label1, label2, entropy)
    if (label1_count == 0 or label2_count == 0) or (len(data[0])==1) or depth >= max_depth:
        return Node(None,None,None,vote, attr_val,depth, 'leaf', label1, label1_count, label2, label2_count)

    flg = True
    for key in result:
        if result[key][2] != 0 and result[key][4] != 0:
            flg = False
    if flg:
        return Node(None,None,None,vote,attr_val,depth, 'leaf', label1, label1_count, label2, label2_count)

    else:
        max_attr, max_idx = get_attr_to_split_on(result)
        node = Node(None, None, max_attr, vote,attr_val,depth, 'internal', label1, label1_count, label2, label2_count)
        left_data = np.delete(data, np.where(data[:, max_idx] == result[max_attr][3])[0], 0)
        right_data = np.delete(data, np.where(data[:, max_idx] == result[max_attr][1])[0], 0)
        left_data = np.delete(left_data, max_idx, 1)
        right_data = np.delete(right_data, max_idx, 1)
        left = train_decision_tree_helper(left_data,result[max_attr][1], node.depth+1,max_depth)
        right = train_decision_tree_helper(right_data,result[max_attr][3], node.depth+1,max_depth)
        node.left = left
        node.right = right
        return node

def printTree(node,  label1, label2, attr='',attr_val=''):
    if node != None:
        if node.attr != None:
            attr = node.attr
        level = node.depth + 1
        if node.type == 'root' or node.type == 'internal':
            if node.left != None:
                attr_val = node.left.attr_val
                label1_count = str(node.left.label1_count)
                node_label1 = node.left.label1
                label2_count = str(node.left.label2_count)
                node_label2 =  node.left.label2
                if node.left.label1_count == 0:
                    node_label1 = label1 if node_label2 != label1 else label2
                elif node.left.label2_count == 0:
                    node_label2 = label2 if node_label1 != label2 else label1
            elif node.attr_val != None:
                attr_val = node.attr_val
            if node_label1 <= node_label2:
                print(' | ' * level + attr +' = ' + attr_val + ": " + "[" + label1_count + " " + node_label1 + "/" + label2_count + " " + node_label2 + "]")
            else:
                print(
                    ' | ' * level + attr + ' = ' + attr_val + ": " + "[" + label2_count + " " + node_label2 + "/" + label1_count + " " + node_label1 + "]")
        printTree(node.left, label1, label2, attr)

        if node.type == 'root' or node.type == 'internal':
            if node.right != None:
                attr_val = node.right.attr_val
                label1_count = str(node.right.label1_count)
                node_label1 = node.right.label1
                label2_count = str(node.right.label2_count)
                node_label2 =  node.right.label2
                if node.right.label1_count == 0:
                    node_label1 = label1 if node_label2 != label1 else label2
                elif node.right.label2_count == 0:
                    node_label2 = label2 if node_label1 != label2 else label1
            elif node.attr_val != None:
                attr_val = node.attr_val
            if node_label1 <= node_label2:
                print(' | ' * level + attr + ' = ' + attr_val + ": "+ "[" + label1_count + " " + node_label1 + "/" + label2_count+ " " + node_label2+ "]")
            else:
                print(
                    ' | ' * level + attr + ' = ' + attr_val + ": " + "[" + label2_count + " " + node_label2 + "/" + label1_count + " " + node_label1 + "]")
        printTree(node.right, label1, label2, attr)

def test_tree(node,row,test_data,attr=''):
    if node != None:
        if node.attr != None:
            attr = node.attr
        idx = -sys.maxsize
        for i in range(0,len(test_data[0])-1):
            if attr == test_data[0][i]:
                idx = i
                break
        attr_val = test_data[row][idx]

        # base case
        if node.left is None and node.right is None and node.attr_val == attr_val:
            return node.vote
        # recursive case
        else:
            if node.left.attr_val == attr_val:
                vote = test_tree(node.left,row,test_data,attr)
            else:
                vote = test_tree(node.right,row,test_data,attr)
            return vote

def predict(root,test_data):
    result_label = []
    for i in range(1,len(test_data)):
        vote = test_tree(root,i,test_data,'')
        result_label.append(vote)
    return result_label

def get_error_rate(label,data):
    right = 0
    wrong = 0
    for i in range(0,len(label)):
        if label[i] == data[i+1][-1]:
            right += 1
        else:
            wrong += 1
    return wrong / (right + wrong)

def write_labels(path, labels):
    '''
    write the label files
    :param data: np array of data
    :param path: output path
    :param majority_label: most common label
    :return: None
    '''
    with open(path, mode='w') as file:
        for label in labels:
            file.write(label + '\n')
    file.close()

def write_metrics(error_train, error_test, metrics):
    '''
    write the metrics file
    :param error_train: error rate of training set
    :param error_test: error rate of testing set
    :param metrics: path to the metrics file
    :return: None
    '''
    with open(metrics, mode='w') as file:
        file.write("error(train): " + str(error_train) + '\n')
        file.write("error(test): " + str(error_test))

if __name__ == '__main__':
    # get the file path from sys argv
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    max_depth = int(sys.argv[3])

    train_labels_path = sys.argv[4]
    test_labels_path = sys.argv[5]
    metrics = sys.argv[6]

    # read in training data
    train_data = read_in_file(train_path)

    # read in test data
    test_data = read_in_file(test_path)

    # get labels
    label1,label1_count, label2, label2_count = count_label(train_data)

    # train the decision tree
    if max_depth > 0:
        root = train_decision_tree(train_data,max_depth)
        # predict on training data
        train_label = predict(root, train_data)

        # predict on test data
        test_label = predict(root, test_data)

        # get training error rate
        train_error = get_error_rate(train_label, train_data)

        # get testing error rate
        test_error = get_error_rate(test_label, test_data)
    else:
        MAJORITY_LABEL = majority_vote(label1, label1_count,label2,label2_count)
        root = Node(None, None, None, MAJORITY_LABEL, None, 0, 'leaf', label1, label1_count, label2, label2_count)
        train_label = []
        for i in range(len(train_data)-1):
            train_label.append(MAJORITY_LABEL)
        test_label = []
        for i in range(len(test_data)-1):
            test_label.append(MAJORITY_LABEL)
        # get training error rate
        train_error = get_error_rate(train_label, train_data)

        # get testing error rate
        test_error = get_error_rate(test_label, test_data)


    # write .labels file
    write_labels(train_labels_path,train_label)
    write_labels(test_labels_path,test_label)
    # write metrics
    write_metrics(train_error,test_error,metrics)

    # print tree
    if label1 <= label2:
        print("[" + str(label1_count) + " " + label1 + "/" + str(label2_count) + " " + label2 + "]")
    else:
        print("[" + str(label2_count) + " " + label2 + "/" + str(label1_count) + " " + label1 + "]")
    printTree(root, label1, label2)
