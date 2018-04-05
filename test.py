import numpy as np
import csv

csvFile = open('train.csv', 'r')
reader = csv.reader(csvFile)
data = []
for i in reader:
    data.append(i)
a = data[0]
b = data[1]
c = data[2]

