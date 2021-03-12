import sys
import math
import numpy as np

def is_number(elem):
    if (elem == ''):
        return False
    n_dot = 0
    first = 1
    for i in str(elem):
        if (i == '.'):
            if (n_dot == 0):
                n_dot = 1
                continue
            else:
                return False
        if (i == '-'):
            if (first == 0):
                return False
            else:
                continue
        if (i.isdigit() == False):
            return False
        first = 0
    return True

def average(feature):
    sumValue = 0
    for value in feature:
        sumValue += value
    return sumValue / len(feature)


def standarDeviation(feature): # = numpy.std
    tmp = feature[:]
    mean = average(tmp)
    for i in range(len(tmp)):
        tmp[i] = abs(tmp[i] - mean)
        tmp[i] *= tmp[i]
    return math.sqrt(average(tmp))
"""
def standarDeviation(feature): #Ecart type
    tmp = feature[:]
    mean = average(tmp)
    for i in range(len(tmp)):
        tmp[i] = abs(tmp[i] - mean)
    return average(tmp)
"""
def minValue(feature):
    tmp = feature[0]
    for value in feature:
        if (value < tmp):
            tmp = value
    return tmp

def quartile(feature, n):
    length = len(feature)
    for i in range(length):
        if (i > n * length / 4):
            if (i == 1):
                return feature[0]
            else:
                return feature[i - 2]
    return feature

def maxValue(feature):
    tmp = feature[0]
    for value in feature:
        if (value > tmp):
            tmp = value
    return tmp

def printValue(value):
    if (is_number(value)):
        print("%15.3f"%value, end="")
    else:
        print("%15s"%value, end="")

    #Main code#

# Protection
if len(sys.argv) != 2:
	print("1 arguments needed: dataset")
	exit(1)

#Open dataset and get lines
dataset_file = open(sys.argv[1], "r")
lines = dataset_file.read().split('\n')

#Save dataset by students
features = lines[0].split(",")
del lines[0]
del lines[-1]
#del features[-12:]
dataset = []
for i in lines:
    dataset.append(i.split(","))

#Save data by features and sort them
data = []
for i in range(len(features)):
    tmp = []
    for student in dataset:
        if (student[i] != ''):
            if (is_number(student[i])):
                tmp.append(float(student[i]))
            else:
                tmp.append(student[i])
    tmp.sort()
    data.append(tmp)

#Use data
count = []
mean = []
std = []
mini = []
Q1 = []
Q2 = []
Q3 = []
maxi = []
extent = []
for feature in data:
    count.append(len(feature))
    mini.append(minValue(feature))
    Q1.append(quartile(feature, 1))
    Q2.append(quartile(feature, 2))
    Q3.append(quartile(feature, 3))
    maxi.append(maxValue(feature))
    if (is_number(feature[0])):
        mean.append(average(feature))
        std.append(standarDeviation(feature))
        extent.append(maxi[-1] - mini[-1])
    else:
        mean.append(np.nan)
        std.append(np.nan)
        extent.append(np.nan)

#Find max length features string
maxLengthFeature = 0
for feature in features:
    if (len(feature) > maxLengthFeature):
        maxLengthFeature = len(feature)

#Print data
for j in range(maxLengthFeature + 1):
    print(" ", end = "")

printValue("count")
printValue("mean")
printValue("std")
printValue("extent")
printValue("min")
printValue("25%")
printValue("50%")
printValue("75%")
printValue("max")

for i in range(len(features)):
    print("\n%s"%features[i], end="")
    for j in range(maxLengthFeature + 1 - len(features[i])):
        print(" ", end = "")
    printValue(count[i])
    printValue(mean[i])
    printValue(std[i])
    printValue(extent[i])
    printValue(mini[i])
    printValue(Q1[i])
    printValue(Q2[i])
    printValue(Q3[i])
    printValue(maxi[i])
