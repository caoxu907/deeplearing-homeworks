import numpy as np
import collections
import pandas as pd

#高斯朴素贝叶斯类
class GuassianNB:
    def __init__(self):
        self.prior = None #先验概率
        self.avgs = None #均值
        self.vars = None #方差值
        self.nums = None # 特征值数量

    #计算先验概率
    def _get_prior(self, label: np.array)->dict:
        cnt = collections.Counter(label)
        a = {}
        for k,v in cnt.items():
            a[k]=v/len(label)
        return a

    #计算均值
    def _get_avgs(self,data:np.array,label:np.array)->np.array:
        return np.array([data[label == i].mean(axis=0) for i in self.nums])

    #计算方差
    def _get_vars(self,data:np.array,label:np.array)->np.array:
        return np.array([data[label == i].var(axis=0) for i in self.nums])

    #计算似然度
    def _get_likelihood(self,row:np.array)->np.array:
        return (1 / np.sqrt(2 * np.pi * self.vars) * np.exp(
            -(row - self.avgs) ** 2 / (2 * self.vars))).prod(axis=1)

    #训练数据集
    def fit(self, data: np.array, label: np.array):
        self.prior = self._get_prior(label)
        print(self.prior)
        a=[]
        for key in self.prior.keys():
            a.append(key)
        self.nums = a
        self.avgs = self._get_avgs(data, label)
        self.vars = self._get_vars(data, label)

    #预测label
    def predict_prob(self, data: np.array) -> np.array:
        likelihood = np.apply_along_axis(self._get_likelihood, axis=1, arr=data)
        #print(likelihood)
        a = []
        for key in self.prior.keys():
            a.append(self.prior[key])
        probs = np.array(a) * likelihood
        #print(probs)
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None]

    #预测结果
    def predict(self, data: np.array) -> np.array:
        return self.predict_prob(data).argmax(axis=1)

def main():
    path = r'D:\hello pytorch\deeplearing\data\data1.csv'
    data = pd.read_csv(path)
    origin_dataset = np.array(data)
    np.random.shuffle(origin_dataset)
    feature_dataset = [] #特征集
    label_dataset = [] #标签集
    feature_dataset_test = [] #测试集
    label_dataset_test = [] #测试结果集
    for i in range(int(len(origin_dataset)*0.8)):
        feature_dataset.append(origin_dataset[i][0:25])
        label_dataset.append(origin_dataset[i][25])
    for i in range(int(len(origin_dataset)*0.8),len(origin_dataset)):
        feature_dataset_test.append(origin_dataset[i][0:25])
        label_dataset_test.append(origin_dataset[i][25])
    feature_dataset = np.array(feature_dataset,dtype=float)
    label_dataset = np.array(label_dataset,dtype=str)
    feature_dataset_test = np.array(feature_dataset_test,dtype=float)
    label_dataset_test = np.array(label_dataset_test,dtype=str)
    nb = GuassianNB()
    nb.fit(feature_dataset,label_dataset) #训练模型
    test_result = nb.predict(feature_dataset_test)

    acc = 0
    for i in range(len(test_result)):
        if nb.nums[test_result[i]] == label_dataset_test[i]:
            acc+=1
    print("精确度："+str(acc/len(test_result)))
if __name__ == '__main__':
    main()
