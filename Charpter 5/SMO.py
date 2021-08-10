from numpy import *
import os


# 先定义一些辅助函数
# 选取第二变量函数
def select_J_rand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


# 定义对α进行裁剪的函数
def clip_alpha(aj, H, L):
     if aj > H:
         aj=H
     if L > aj:
         aj = L
     return aj


def smo_simple(dataX, dataY, C, toler, iter_num):
    dataMatrix = mat(dataX); labelMat = dataY.transpose()
    # 初始化参数
    b = 0; m, n = np.shape(dataMatrix)
    alphas = mat(np.zeros((m, 1)))
    iter = 0
    # 当超过迭代次数停止
    while iter < iter_num:
        # 记录α1和α2变化次数
        alphaPairsChanged = 0
        for i in range(m):
            # 计算f(xi),预测类别
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 计算误差
            Ei = fXi - float(labelMat[i])
            # 当不满足条件时，选取变量j，这里要判断α是否在[0,C]内，若超出范围则不再进行优化
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and alphas[i] > 0):
                j = select_J_rand(i, m)
                # 计算x2的预测值y2
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alpha_I_old = alphas[i].copy()
                alpha_J_old = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H:
                    print("L == H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta > 0:
                    print("eta > 0")
                    continue
                alphas[j] -= labelMat[j] * (Ej - Ei)/eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                # 当α2不再变化
                if (abs(alphas[j]-alpha_J_old) < 0.00001):
                    print("j not moving enough")
                    continue
                # 更新α1　　　　
                alphas[i] += labelMat[i] * labelMat[j] * (alpha_J_old - alphas[j])
                # 计算b1和b2　　　　　　
                b1 = b - Ei - labelMat[i] * (alphas[i] - alpha_I_old) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alpha_J_old) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alpha_I_old) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alpha_J_old) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (alphas[i] > 0) and (alphas[i] < C):
                    b = b1
                elif (alphas[j] > 0) and (alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2)/2
                alphaPairsChanged += 1
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d"%iter)
    return b, alphas


# 首先建立3个辅助函数用于对E进行缓存，以及1种用于清理代码的数据结构
# 存储变量的数据结构
class optStruct:
    def __init__(self, dataX, dataY, C, toler, kTup):
        self.X = dataX
        self.Y = dataY
        self.C = C
        self.toler = toler
        self.m = np.shape(dataX)[0]
        self.alphas = np.zeros((self.m, 1))
        self.b = 0
        # cache第一列为有效性标志位，第二列为E值
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = mat(zeros(self.m, self.m))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], self.kTup)

# 计算E值并返回，由于频繁使用单独写成一个函数
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 用于选择第二个α的值，保证每次优化采用最大的步长
def select_J(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.Cache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # 选择变化最大的那个
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = select_J_rand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 与simpleSMO一致，更新的alpha存入cache中
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.toler) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.toler) and (oS.alphas[i] > 0)):
        j, Ej = select_J(i, oS, Ei)
        alpha_I_old = oS.alphas[i].copy()
        alpha_J_old = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C+ oS.alphas[j] - oS.alphas[i])
        else:
            L = min(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = max(oS.C, oS.alphas[i] + oS.alphas[j])
        if H == L:
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i], - oS.K[j, j]
        if eta >= 0:
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alpha_J_old) < 0.00001:
            return 0
        oS.alphas[i] -= oS.labelMat[i] * oS.labelMat[j] * (oS.alphas[j] - alpha_J_old)
        updateEk(oS, i)

        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alpha_I_old) * oS.K[i, i] - oS.labelMat[j] * (
                    oS.alphas[j] - alpha_J_old) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alpha_I_old) * oS.K[i, j] - oS.labelMat[j] * (
                    oS.alphas[j] - alpha_J_old) * oS.K[j, j]
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
            oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            os.b = (b1 + b2)/2
        return 1
    else:
        return 0


def smoP(dataX, labelMat, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataX), mat(labelMat).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and alphaPairsChanged > 0 or entireSet:
        alphaPairsChanged = 0
        # 第一种遍历全体样本
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                iter += 1
        # 第二种遍历非边界样本
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * oS.alphas.A < C)[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
    return oS.alphas, oS.b


def calW(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((m, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :])
    return w


# 首先建立计算核函数转换函数
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros(m, 1))
    if kTup['0'] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K/(-1 * kTup[1] ** 2))
    else:
        raise NameError('没有定义核函数')
    return K


def img2Vector(filename):
    returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32*i + j] = int(lineStr[j])
    return returnVec


def loadImages(dir):
    hwLabels = []
    trainingFileList = os.listdir(dir)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2Vector('%s/%s' % (dir, fileNameStr))
    return trainingMat, hwLabels


dir = '.\data\\trainingDigits'


def predict(dataArr, labelArr, alphas, b, kTup):
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    svs = dataMat[svInd]
    labelSv = labelMat[svInd]
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(svs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSv, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    return svInd, labelSv, svs, errorCount / (len(labelMat))


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('F:\自学2020\PythonML_Code\Charpter 5\data\\trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    svs = dataMat[svInd]
    labelSv = labelMat[svInd]
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(svs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSv, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('there are %d Support Vectors' % shape(svs)[0])
    print('the error rate is %f' % (errorCount / (len(labelMat))))
    test_dataArr, test_labelArr = loadImages('.\data\\testDigits')
    test_dataMat = mat(test_dataArr)
    test_labelMat = mat(test_labelArr).transpose()
    m1, n1 = shape(test_dataMat)
    test_errorCount = 0
    for i in range(m1):
        kernelEval = kernelTrans(svs, test_dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSv, alphas[svInd]) + b
        if sign(predict) != sign(test_labelArr[i]):
            errorCount += 1
    print('the error rate is %f' % (test_errorCount / (len(test_labelMat))))











