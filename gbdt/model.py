# -*- coding:utf-8 -*-
from datetime import datetime
from random import sample
from math import exp, log
from gbdt.tree import construct_decision_tree


class GBDT:
    def __init__(self, max_iter, sample_rate, learn_rate, max_depth, split_points=0):
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.split_points = split_points
        self.trees = dict()

    def train(self, dataset, train_data, stat_file, test_data=None):
        label_valueset = dataset.get_label_valueset()
        f = dict()  # for train instances
        self.initialize(f, dataset)
        for iter in range(1, self.max_iter+1):
            subset = train_data
            if 0 < self.sample_rate < 1:
                subset = sample(subset, int(len(subset)*self.sample_rate))
            self.trees[iter] = dict()
            # 用损失函数的负梯度作为回归问题提升树的残差近似值
            residual = self.compute_residual(dataset, subset, f)
            # print "resiudal of iteration",iter,"###",residual;
            for label in label_valueset:
                # 挂叶子节点的下的各种样本,只有到迭代的max-depth才会使用...
                # 存放的各个叶子节点，注意叶子节点存放的是各个条件下的样本集点
                leafNodes = []
                targets = {}
                for id in subset:
                    targets[id] = residual[id][label]
                # for debug
                # print "targets of iteration:",iter,"and label=",label,"###",targets;
                # construct_decision_tree(dataset,remainedSet,targets,depth,leafNodes,max_depth,split_points=0):
                # 对某一个具体的label-K分类，选择max-depth个特征构造决策树
                tree = construct_decision_tree(dataset, subset, targets, 0, leafNodes, self.max_depth, self.split_points)
                # if label==sample(label_valueset,1)[0]:
                # print tree.describe("#"*30+"Tree Description"+"#"*30+"\n");
                self.trees[iter][label] = tree
                self.update_f_value(f, tree, leafNodes, subset, dataset, label)
            # for debug
            #print "residual=",residual;
            if test_data:
                accuracy, ave_risk = self.test(dataset, test_data)
            train_loss = self.compute_loss(dataset, train_data, f)
            test_loss = self.compute_loss(dataset, test_data, f)
            stat_file.write(str(iter)+"\t"+str(train_loss)+"\t"+str(accuracy)+"\t"+str(test_loss)+"\n")
            if iter % 1 == 0:
                print("accuracy=%f,average train_loss=%f,average test_loss=%f" % (accuracy,train_loss,test_loss))
                label = "+"
                print("stop iteration:", iter, "time now:", datetime.now())
                print("\n")

    def compute_loss(self, dataset, subset, f):
        loss = 0.0
        for id in subset:
            instance = dataset.get_instance(id)
            if id in f:
                f_values = f[id]
            else:
                f_values = self.compute_instance_f_value(instance, dataset.get_label_valueset())
            exp_values = {}
            for label in f_values:
                exp_values[label] = exp(f_values[label])
            probs = {}
            for label in f_values:
                probs[label] = exp_values[label]/sum(exp_values.values())
                # 预测的越准确则log(probs[instance["label"]])越接近0 loss也就越小
            loss -= log(probs[instance["label"]])
        return loss/len(subset)

    def initialize(self, f, dataset):
        for id in dataset.get_instances_idset():
            f[id] = dict()
            for label in dataset.get_label_valueset():
                f[id][label] = 0.0

    def update_f_value(self, f, tree, leafNodes, subset, dataset, label):
        data_idset = set(dataset.get_instances_idset())
        subset = set(subset)
        for node in leafNodes:
            for id in node.get_idset():
                f[id][label] = f[id][label]+self.learn_rate*node.get_predict_value()
        # for id not in subset, we have to predict by retrive the tree
        for id in data_idset-subset:
            f[id][label] = f[id][label]+self.learn_rate*tree.get_predict_value(dataset.get_instance(id))

    # subset是从训练集中抽样出来的部分数据...
    def compute_residual(self, dataset, subset, f):
        residual = {}
        label_valueset = dataset.get_label_valueset()
        for id in subset:
            residual[id] = {}
            p_sum = sum([exp(f[id][x]) for x in label_valueset])
            # 对于同一样本在不同类别的残差，需要在同一次迭代中更新在不同类别的残差
            for label in label_valueset:
                p = exp(f[id][label])/p_sum
                y = 0.0
                if dataset.get_instance(id)["label"] == label:
                    y = 1.0
                residual[id][label] = y-p
        return residual

    # 在进行预测的时候，得到某一个具体的instance样本点在K个分类下的浮点值
    def compute_instance_f_value(self, instance, label_valueset):
        f_value=dict()
        for label in label_valueset:
            f_value[label]=0.0
        for iter in self.trees:
            for label in label_valueset:
                tree=self.trees[iter][label]
                f_value[label]=f_value[label]+self.learn_rate*tree.get_predict_value(instance)
        return f_value

    def test(self, dataset, test_data):
        right_predition = 0
        label_valueset = dataset.get_label_valueset()
        risk = 0.0
        for id in test_data:
            instance = dataset.get_instance(id)
            predict_label, probs = self.predict_label(instance, label_valueset)
            single_risk = 0.0
            for label in probs:
                if label == instance["label"]:
                    single_risk += 1.0-probs[label]
                else:
                    single_risk = single_risk+probs[label]
            # print probs,"instance label=",instance["label"],"##single_risk=",single_risk/len(probs);
            risk += single_risk/len(probs)
            if instance["label"] == predict_label:
                right_predition += 1
        # print "test data size=%d,test accuracy=%f"%(len(test_data),float(right_predition)/len(test_data));
        return float(right_predition)/len(test_data), risk/len(test_data)

    # 返回预测的分类K，及K个分类各自的概率值prob list
    def predict_label(self, instance, label_valueset):
        f_value = self.compute_instance_f_value(instance, label_valueset)
        predict_label = None
        exp_values = dict()
        for label in f_value:
            exp_values[label] = exp(f_value[label])
        exp_sum = sum(exp_values.values())
        probs = dict()
        # 归一化，并得到相应的概率值
        for label in exp_values:
            probs[label] = exp_values[label]/exp_sum

        # 选出K分类中，概率值最大的label，返回预测分类的label概率值list
        for label in probs:
            if not predict_label or probs[label] > probs[predict_label]:
                predict_label = label
        return predict_label, probs
