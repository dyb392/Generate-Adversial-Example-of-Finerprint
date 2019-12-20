import csv
from fingerprint import main
import numpy as np

if __name__ == "__main__":
    test = []
    with open("./test_common.csv", "r", newline="") as r:
        reader = csv.reader(r)
        line = 0
        for row in reader:
            if line == 1:
                test.append(row)
            line = 1
    #print(np.array(test).shape)
    train = []
    with open("./train_common.csv", "r", newline="") as r:
        reader = csv.reader(r)
        line = 0
        for row in reader:
            if line == 1:
                train.append(row)
            line = 1
    #print(train)    
    #dbclass = 5
    classNum = 10
    trainNum = 1
    testNum = 7
    right = 0
    #for db in range(dbclass):
    for index in range(classNum):
        trainImage = train[index][0]
        for te in range(testNum):
            testImage = test[testNum*index+te][0]
            image = []
            image.append(trainImage)
            image.append(testImage)
            result = main(image)
            if result == 1:
                right+=1
            print(index,te)
    print(right)
    print(right/len(test))
            
