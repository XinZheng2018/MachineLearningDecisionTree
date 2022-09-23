# Spring 2022 10601 HW2
# Author: Xin Zheng

import numpy as np
import sys
import math

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

def train_data(label1, label1_count, label2, label2_count):
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

def get_error_rate(majority_label, data):
    '''
    get the ratio of the number of errors to the total number test
    :param majority_label: most common label
    :param data: np array of data
    :return: the error rate
    '''
    error = 0
    for i in range(1, len(data)):
        row = data[i]
        if row[-1] != majority_label:
            error += 1
    return error/(len(data)-1)

def get_entropy(label1_count,label2_count):
    '''
    Calculate the entropy before splitting
    :param label1: first label
    :param label1_count: number of label1
    :param label2: second label
    :param label2_count: number of label2
    :return: the entropy of the set
    '''
    total_num = label1_count + label2_count
    return -(label1_count/total_num)*math.log2(label1_count/total_num)-(label2_count/total_num)*math.log2(label2_count/total_num)

def write_output(output_path, entropy, error_rate):
    '''
    Write the output file for inspection
    :param output_path: path to the output file
    :param entropy: value of entropy
    :param error_rate: value of error rate
    '''
    with open(output_path, mode='w') as file:
        file.write("entropy: " + str(entropy) + '\n')
        file.write("error: " + str(error_rate))

if __name__=='__main__':
    # get the file path from sys argv
    train_path = sys.argv[1]
    output_path = sys.argv[2]

    # read in training file

    train_set = read_in_file(train_path)

    # get labels and label counts
    label1, label1_count, label2, label2_count = count_label(train_set)

    # get entropy
    entropy = get_entropy(label1_count, label2_count)

    # get the majority label
    MAJORITY_LABEL = train_data(label1, label1_count, label2, label2_count)

    # compute the error rate
    error_rate = get_error_rate(MAJORITY_LABEL, train_set)

    # produce output file
    write_output(output_path, entropy, error_rate)