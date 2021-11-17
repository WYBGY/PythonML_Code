from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def lwlr(testpoint, x, y, k=1.0):
    x_mat = mat(x)
    y_mat = mat(y).T
    m = shape(x)[0]
    weights = mat(eye(m))
    for i in range(m):
        diff_ = testpoint - x_mat[i, :]
        weights[i, i] = exp(diff_ * diff_.T/(-2 * k ** 2))

    xTx = x_mat.T * (weights * x_mat)
    if linalg.det(xTx) == 0:
        return
    ws = xTx.I * (x_mat.T * (weights * y_mat))
    return testpoint * ws


def lwlrTest(testArr, x, y, k):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], x, y, k)
    return yHat


x, y = loadDataSet('F:\自学2020\PythonML_Code\Charpter 1/ex0.txt')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(mat(x)[:, 1].flatten().A[0], mat(y).T[:, 0].flatten().A[0])

yHat = lwlrTest(x, x, y, 0.03)
srtind = mat(x)[:, 1].argsort(0)
xsort = mat(x)[srtind][:, 0, :]
ax.plot(xsort[:, 1], yHat[srtind], 'red', linewidth=2)






