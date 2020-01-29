# # Processing Data
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR


data = pd.read_csv('hw5p6data.csv')

label = np.array([x[0] for x in data.values])
feature = [x[1:] for x in data.values]

# affine feature expansion
affine_f = np.zeros((len(feature), 1), np.int8)
affine_f.fill(1)
affine_f = affine_f.tolist()

for i in range(len(feature)):
    affine_f[i].append(feature[i][0])
    affine_f[i].append(feature[i][1])

affine_f = np.array(affine_f)


# (a)

model = LR(solver='lbfgs')
model.fit(affine_f, label)


predict_l = model.predict(affine_f)


def error_rate(predict, label):
    count = 0
    for i in range(predict.shape[0]):
        if predict[i] != label[i]:
            count += 1
    return count / predict.shape[0]

# get the training error rate
train_error = error_rate(predict_l, label)
print(train_error)

# get the MLE w hat
w = model.coef_
print(w)


# (b)

print(error_rate(predict_l[0:500], label[0:500])) 
print(error_rate(predict_l[500:], label[500:]))


# (c)

def positive_rate(pl, total):
    count = 0.0
    for x in pl:
        if x == 1:
            count += 1
    return count/total

print('classiﬁcation discrepancy:')
print(positive_rate(predict_l[0:500], 500) - positive_rate(predict_l[500:], 100))


# (d)

def fnr(pl,l):
    positive = 0.0
    l_wanted = []
    for i in range(pl.shape[0]):
        if pl[i] == 1:
            positive += 1
            l_wanted.append(l[i])
    negative = 0.0
    for x in l_wanted:
        if x == -1:
            negative += 1
    return negative/positive

print('FNR discrepancy')
print(fnr(predict_l[0:500], label[0:500]) - fnr(predict_l[500:], label[500:]))

