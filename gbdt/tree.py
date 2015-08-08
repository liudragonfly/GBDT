# -*- coding:utf-8 -*-
from math import log
from random import sample


class Tree:
    def __init__(self):
        self.split_feature = None
        self.leftTree = None
        self.rightTree = None
        # condition for real value is < , for category value is =
        # is for the left-path tree
        self.real_value_feature = True
        self.conditionValue = None
        self.leafNode = None

    def get_predict_value(self, instance):
        if self.leafNode:  # we are in the leaf node
            return self.leafNode.get_predict_value()
        if not self.split_feature:
            raise ValueError("the tree is null")
        if self.real_value_feature and instance[self.split_feature] < self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        elif not self.real_value_feature and instance[self.split_feature] == self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        return self.rightTree.get_predict_value(instance)

    def describe(self, addtion_info=""):
        if not self.leftTree or not self.rightTree:
            return self.leafNode.describe()
        leftInfo = self.leftTree.describe()
        rightInfo = self.rightTree.describe()
        info = addtion_info+"{split_feature:"+str(self.split_feature)+",split_value:"+str(self.conditionValue)+"[left_tree:"+leftInfo+",right_tree:"+rightInfo+"]}"
        return info


class LeafNode:
    def __init__(self, idset):
        self.idset = idset
        self.predictValue = None

    def describe(self):
        return "{LeafNode:"+str(self.predictValue)+"}"

    def get_idset(self):
        return self.idset

    def get_predict_value(self):
        return self.predictValue

    def update_predict_value(self, targets, K):  # K is number of class, just for classification
        sum1 = sum([targets[x] for x in self.idset])
        sum2 = sum([abs(targets[x])*(1.0-abs(targets[x])) for x in self.idset])
        if sum1 == 0:
            self.predictValue = 0
        else:
            try:
                self.predictValue = float(K-1)/K*(sum1/sum2)
            except ZeroDivisionError:
                print("zero division,sum1=%f,sum2=%f" % (sum1, sum2))
                print("targets are:", [targets[x] for x in self.idset])
                raise


def MSE(values):
    if len(values) < 2:
        return 0
    mean = sum(values)/float(len(values))
    error = 0.0
    for v in values:
        error += (mean-v)*(mean-v)
    return error


def FriedmanMSE(left_values, right_values):
    # 假定每个样本的权重都为1
    weighted_n_left, weighted_n_right = len(left_values), len(right_values)
    total_meal_left, total_meal_right = sum(left_values)/float(weighted_n_left), sum(right_values)/float(weighted_n_right)
    diff = total_meal_left - total_meal_right
    return (weighted_n_left * weighted_n_right * diff * diff /
            (weighted_n_left + weighted_n_right))


def compute_min_loss(values):
    if len(values) < 2:
        return 0
    mean = sum(values)/float(len(values))
    loss = 0.0
    for v in values:
        loss += (mean - v) * (mean - v)
    return loss


# if split_points is larger than 0, we just random choice split_points to evalute minLoss
# when consider real-value split
def construct_decision_tree(dataset, remainedSet, targets, depth, leafNodes, max_depth, split_points=0):
    # print "start process,depth=",depth;
    if depth < max_depth:
        # 通过修改这里可以实现max_features的指定
        attributes = dataset.get_attributes()
        loss = -1
        selectedAttribute = None
        conditionValue = None
        selectedLeftIdSet = []
        selectedRightIdSet = []
        for attribute in attributes:
            # print "start process attribute=",attribute;
            is_real_type = dataset.is_real_type_field(attribute)
            attrValues = dataset.get_distinct_valueset(attribute)
            if is_real_type and split_points > 0 and len(attrValues) > split_points:  # need subsample split points to speed up
                attrValues = sample(attrValues, split_points)
            for attrValue in attrValues:
                leftIdSet = []
                rightIdSet = []
                for Id in remainedSet:
                    instance = dataset.get_instance(Id)
                    value = instance[attribute]
                    if (is_real_type and value < attrValue)or(not is_real_type and value == attrValue):   # fall into the left
                        leftIdSet.append(Id)
                    else:
                        rightIdSet.append(Id)
                leftTargets = [targets[id] for id in leftIdSet]
                rightTargets = [targets[id] for id in rightIdSet]
                sumLoss = compute_min_loss(leftTargets)+compute_min_loss(rightTargets)
                if loss < 0 or sumLoss < loss:
                    selectedAttribute = attribute
                    conditionValue = attrValue
                    loss = sumLoss
                    selectedLeftIdSet = leftIdSet
                    selectedRightIdSet = rightIdSet
            # print "for attribute:",attribute," min loss=",loss
        # print "process over, get split attribute=",selectedAttribute
        if not selectedAttribute or loss < 0:
            raise ValueError("cannot determine the split attribute.")
        tree = Tree()
        tree.split_feature = selectedAttribute
        tree.real_value_feature = dataset.is_real_type_field(selectedAttribute)
        tree.conditionValue = conditionValue
        tree.leftTree = construct_decision_tree(dataset, selectedLeftIdSet, targets, depth+1, leafNodes, max_depth)
        tree.rightTree = construct_decision_tree(dataset, selectedRightIdSet, targets, depth+1, leafNodes, max_depth)
        # print "build a tree,min loss=",loss,"conditon value=",conditionValue,"attribute=",tree.split_feature
        return tree
    else:  # is a leaf node
        node = LeafNode(remainedSet)
        K = dataset.get_label_size()
        node.update_predict_value(targets, K)
        leafNodes.append(node)  # add a leaf node
        tree = Tree()
        tree.leafNode = node
        return tree
