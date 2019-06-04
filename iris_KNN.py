import numpy as np
from sklearn.model_selection import train_test_split
#引入划分训练集和测试集的工具
from sklearn import neighbors
#引入KNN算法


#划分训练集和测试集
iris_matrix=np.loadtxt(open(r'C:\Users\asus\Desktop\iris\iris_database.csv', 'rt'),delimiter=",",skiprows=1)
#建立一个150*5的矩阵，采样跳过第一行
X, y = iris_matrix[:,:-1],iris_matrix[:,-1]
#特征值和标签的取值
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)
#30%作为测试集，rm控制随机划分
train= np.column_stack((X_train,y_train))
#np.savetxt('iris_train.csv',train, delimiter = ',')
test = np.column_stack((X_test, y_test))
#np.savetxt('iris_test.csv', test, delimiter = ',')
#保存测试集和训练集


#KNN的实现
KNN = neighbors.KNeighborsClassifier()
#引入算法
KNN.fit(X_train,y_train)
#用数据训练
iris_y_predict = KNN.predict(X_test) 
#预测
score=KNN.score(X_test,y_test,sample_weight=None)
#计算出准确率
probility=KNN.predict_proba(X_test)  
 #计算各测试样本基于概率的预测（贝叶斯流派）
print(iris_y_predict)
#预测结果
print(score)
#准确率
print(probility)
#贝叶斯概率[第i行j列表示第i个样本是标签j的概率]