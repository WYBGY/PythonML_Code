def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


#
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItem = len(D)
    retList = []
    supportData = {}
    for key in ssCnt:
        support = float(ssCnt[key] / numItem)
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def apriori_gen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
                print(i, j, L1, L2)
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = dataSet
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = apriori_gen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


# apriori计算从A→B的支持度
# 生成规则
def generateRules(L, supportData, minConf=0.5):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H1, supportData, brl, minConf=0.5):
    pruneH = []
    for conseq in H1:
        conf = supportData[freqSet]/supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '--->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            pruneH.append(conseq)
    return pruneH


def rulesFromConseq(freqSet, H1, supportData, brl, minConf):
    m = len(H1[0])
    if len(freqSet) > m + 1:
        Hmp1 = apriori_gen(H1, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


