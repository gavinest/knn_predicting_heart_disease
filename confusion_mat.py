import numpy as np
import pandas as pd


class ConfusionMat(object):
    '''
    creates object that prints confusion matrix when called.

    INPUT:
        - LIST, ARRAY true values
        - LIST, ARRAY model predicted values
    '''
    def __init__(self, y_true, y_predict):
        self.y_true = y_true
        self.y_predict = y_predict
        self.ys = zip(self.y_true, self.y_predict)
        self.calc_mat()

    def calc_mat(self):
        tp, fp, tn, fn = 0, 0, 0, 0
        for t, p in self.ys:
            if t == p and t == 1:
                tp += 1
            elif t == p and t == 0:
                tn += 1
            elif t != p and t == 1:
                fn += 1
            else:
                #t != p and t == 0:
                fp +=1

        cols = ['Predicted +', 'Predicted -']
        index = ['True +', 'True -']
        self.cm = pd.DataFrame(np.array([[tp, fn], [fp, tn]]), index=index, columns=cols)
        print self.cm




if __name__ == '__main__':
    y_true = [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]
    y_pred = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    ConfusionMat(y_true, y_pred)
