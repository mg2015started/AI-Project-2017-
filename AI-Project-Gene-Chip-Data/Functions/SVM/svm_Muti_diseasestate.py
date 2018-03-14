#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import csv

def setList(index, labelNum):
    l = []
    for i in range (labelNum):
        if i == index:
            l.append(1)
        else:
            l.append(-1)
    return l

def findIndex(list, element):
    for i in list:
        if element == i:
            return list.index(i)
    return -1

def roundRobin(l, turn):
    for i in range(int(len(l)/turn)):
        l.append(l[i])
        del l[i]


if __name__ == '__main__':

    # Be verbose?
    verbose = 1

    # Get the C param of SVM
    svmC = 0.9

    BATCH_SIZE = 100  # The number of training examples to use per training step.

    labelList = []
    labelNum = 79
    numTrain = 5896
    numTest = 5896
    iteration = 100
    robin = 10
    batch = 100
    rate = 0.05
    errorTrain = 0
    errorTest = 0
    trX = []    # 5894x112
    trY = []    # 5894x2
    teX = []
    teY = []
    normal_cnt = 0
    disease_cnt = 0
    invalid_cnt = 0
    num=0
    #train
    fy = open('../../E-TABM-185.sdrf.txt','r')
    fx = open('../../data_dimRed_0.9.txt','r')
    fo = open('output1.txt','w')
    fy.readline()
    errorTrain = 0
    suit_label = []
    finallist = []
    with open('../../labelDetail/8_Characteristics [DiseaseState].csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if int(row[1]) < 10:
                suit_label.append(row[0])

    #print(suit_label)

    for i in range(0, 5896):
        line_x = fx.readline().split('\t')[:-1]
        line_x = list(map(float, line_x))
        # if(i%300 == 0):
        #   print(len(line_x))
        line_y = fy.readline().split('\t')[28:29][0].strip('"')
        if line_y in suit_label:
            continue
        else:
            if line_y[0:2] == "  ":
                invalid_cnt = invalid_cnt + 1
                continue
            else:
                if line_y == 'healthy' or line_y == 'normal':
                    normal_cnt = normal_cnt + 1
                else:
                    disease_cnt = disease_cnt + 1
                ret = findIndex(labelList, line_y)
                if ret == -1:
                    num += 1
                    labelList.append(line_y)
                    line_y = setList(len(labelList) - 1, labelNum)
                else:
                    line_y = setList(ret, labelNum)

            trX.append(line_x)
            trY.append(line_y)

    order = np.random.permutation(len(trX))
    order = order.tolist()
    tpx = [trX[i] for i in order]
    tpy = [trY[i] for i in order]
    trX = tpx
    trY = tpy
    print(num)

        #if(i%300 == 0):
           #print(line_y)

    #for i in range(0,590):
    #   line_x = fx.readline().split('\t')[:-1]
    #   line_x = list(map(float,line_x))
    #   teX.append(line_x)
    #    line_y = fy.readline().split('\t')[1:2][0]
    #    if line_y == 'organism_part':
    #        line_y = [1,0]
    #    else :
    #        line_y = [0,1]
    #    teY.append(line_y)
    print("trx:", len(trX))
    print("try:", len(trY))
    print("dim", len(trX[0]))
    dtlist = []
    for i in range(iteration):
        dtlist.append(0)
    for turn in range(10):
        roundRobin(trX, robin)
        roundRobin(trY, robin)

        teX = trX[0:round(len(trX) / robin)]
        teY = trY[0:round(len(trY) / robin)]
        # print(normal_cnt)
        # print(disease_cnt)
        # print(invalid_cnt)
        fx.close()
        fy.close()

        x = tf.placeholder("float", [None, len(trX[0])])  # create symbolic variables
        y = tf.placeholder("float", [None, labelNum])

        # the classification.
        W = tf.Variable(tf.zeros([len(trX[0]), labelNum]))
        b = tf.Variable(tf.zeros([labelNum]))
        y_raw = tf.matmul(x, W) + b

        # Optimization.
        regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
        hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([batch, labelNum]),
                                              1 - y * y_raw));
        svm_loss = regularization_loss + svmC * hinge_loss;
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

        # Evaluation.
        predicted_class = tf.sign(y_raw);
        correct_prediction = tf.equal(y, predicted_class)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        aclist = []
        # Launch the graph in a session
        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                # you need to initialize all variables
                tf.global_variables_initializer().run()

                for i in range(iteration):
                    for start, end in zip(range(int(len(trX) / robin), len(trX), batch),
                                          range(int(len(trX) / robin) + batch, len(trX) + 1, batch)):
                        sess.run(train_step, feed_dict={x: trX[start:end], y: trY[start:end]})
                    accu = accuracy.eval(feed_dict={x: teX, y: teY})
                    aclist.append(accu)
                    dtlist[i] += accu
                finallist.append(sum(aclist) / len(aclist))
                print(turn, finallist[-1])

    print("avaccuracy:", sum(finallist) / len(finallist))
    dtlist = [item / 10 for item in dtlist]
    for item in dtlist:
        fo.write(str(item) + '\n')
    fo.close()
