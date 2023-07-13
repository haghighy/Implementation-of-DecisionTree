import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import math


class TreeNode:
    def __init__(self, feature=None, amountThr=None,g=None, Entropy=None, Left=None, Right=None,leafVal=None):
        self.feature = feature
        self.amountThr = amountThr
        self.Left = Left
        self.Right = Right
        self.leafVal = leafVal
        self.gain = g
        self.Entropy = Entropy
    def IsLeaf(self):
        return self.leafVal is not None


class DecisionTree:
    def __init__(self, MaxDepth=100, featureCount=None):
        self.MaxDepth=MaxDepth
        self.featureCount=featureCount
        self.root=None

    def fitRoot(self, X, y):
        if self.featureCount == None:
            self.featureCount = X.shape[1]
        else:
            self.featureCount = min(X.shape[1],self.featureCount)
        self.root = self.ExpandTree(X, y)

    def ExpandTree(self, X, y, depth=0):
        sampleCount = X.shape[0]
        featureCount = X.shape[1]
        LableCount = len(np.unique(y))

        if (depth>=self.MaxDepth or LableCount==1 or sampleCount<2):
            c = Counter(y)
            leaf_Val = c.most_common(2)[0][0]
            return TreeNode(leafVal=leaf_Val)

        best_feature, best_thresh, Bestgain, best_entp = self.SplitWithBest(X, y, featureCount)
        # create childs
        leftIdxs, rightIdxs = self.Splitnode(X[:, best_feature], best_thresh)
        Left = self.ExpandTree(X[leftIdxs, :], y[leftIdxs], depth+1)
        Right = self.ExpandTree(X[rightIdxs, :], y[rightIdxs], depth+1)
        return TreeNode(best_feature, best_thresh,Bestgain, best_entp, Left, Right)

    def MyEntropy(self, y):
        ps = np.bincount(y)/ len(y) #[0's count, 1's count]
        array = []
        for p in ps:
            if p>0:
                array.append(p * np.log(p))
        result_Entp = -1 * np.sum(array)
        if np.sum(array) == 0:
            return 0
        return result_Entp

    def SplitWithBest(self, X, y, feat_idxs):
        Bestgain = -1
        split_idx = None
        split_threshold = None
        for feat_idx in range(feat_idxs):
            X_column = X[:, feat_idx]
            amountThrs = np.unique(X_column)
            for thr in amountThrs:
                tmp = self.InformationGain(y, X_column, thr)
                if tmp != 0:
                    gain = tmp[0]
                    entp = tmp[1]
                else:
                    gain = 0
                    entp = 0
                if gain > Bestgain:
                    Bestgain = gain
                    best_entp = entp
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold, Bestgain, best_entp

    def Splitnode(self, X_column, s_amountThrsh):
        l_idx = np.argwhere(X_column <= s_amountThrsh)
        IDXLeft = l_idx.flatten()
        r_idx = np.argwhere(X_column > s_amountThrsh)
        IDXRight = r_idx.flatten()
        return IDXLeft, IDXRight

    def InformationGain(self, y, X_column, amountThr):
        EntropyParent = self.MyEntropy(y)
        leftIdxs, rightIdxs = self.Splitnode(X_column, amountThr)
        if len(leftIdxs) == 0 or len(rightIdxs) == 0:
            return 0
        # weighted avgerage entropy of childs
        y_count = len(y)
        left_count = len(leftIdxs)
        right_count = len(rightIdxs)
        left_entp = self.MyEntropy(y[leftIdxs])
        right_entp = self.MyEntropy(y[rightIdxs])
        childEntropy = (left_count/y_count)*left_entp + (right_count/y_count)*right_entp

        information_gain = EntropyParent - childEntropy
        arr = []
        arr.append(information_gain)
        arr.append(childEntropy)
        return arr

    def TreeTraverse(self, x, node):
        if node.IsLeaf():
            return node.leafVal
        if x[node.feature] <= node.amountThr:
            return self.TreeTraverse(x, node.Left)
        return self.TreeTraverse(x, node.Right)

def Find_DecisionTree (node):
    if node == None:
        return
    if node.IsLeaf():
        print("- leaf\t" , node.leafVal)
    else:
        print("- feat\t" ,node.feature , "  - thresh\t" , node.amountThr, "  - information gain\t", node.gain, "  - entropy\t", node.Entropy)
    Find_DecisionTree(node.Left)
    Find_DecisionTree(node.Right)

dataset = pd.read_csv(".\diabetes.csv")
# dataset = pd.read_csv(".\DataRestaurant.csv")
datelen = len(dataset.columns)-1
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, datelen].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0 #1234
)

# X_train = X
# X_test = []
# y_train = y
# y_test = []

clf = DecisionTree(MaxDepth=50)
clf.fitRoot(X_train, y_train)

#test X_test
ResArray = []
for xx in X_test:
    res = clf.TreeTraverse(xx, clf.root)
    ResArray.append(res)
Result = np.array(ResArray)

#calculate right answers (deghat)
deghat = np.sum(y_test == Result) / len(y_test)
print("deghat(accuracy): ", deghat)

Find_DecisionTree(clf.root)