#谢政的L2作业
import numpy as np
import matplotlib.pyplot as plt
import datetime

#各类自定超参数
iter_num = 2000 #PLA算法的迭代次数
data_size = 200 #数据集的大小

#生成数据所用的均值向量与协方差矩阵
X1_mean = np.array([1,0])
X1_cov = np.array([[1,0],[0,1]])
X2_mean = np.array([0,1])
X2_cov = np.array([[1,0],[0,1]])

#初始化PLA算法与Pocket算法的权重
W0_PLA = np.array([1.0,1.0,1.0])
W0_Pocket = np.array([1.0,1.0,1.0])

#根据均值向量与协方差矩阵生成数据
def GenData(data_size):
    np.random.seed(1)
    X1 = np.random.multivariate_normal(X1_mean,X1_cov,data_size)
    Y1 = np.ones(data_size)
    X2 = np.random.multivariate_normal(X2_mean,X2_cov,data_size)
    Y2 = -np.ones(data_size)
    return X1,Y1,X2,Y2

#增广化X1和X2，为PLA算法作准备
def Augment(X):
    X = np.hstack((np.ones((len(X),1)),X))
    return X

#统计正确率
def Accuracy(X,Y,W):
    correctNum = 0
    for i in range(len(X)):
        if Sign(np.dot(X[i],W)) == Y[i]:
            correctNum += 1
    return correctNum/len(X)

#符号函数
def Sign(x):
    if x>=0:
        return 1
    else:
        return -1
    
#PLA算法,输入样本集X，标签集Y，初始权重W0，迭代次数iter_num
#输出最终权重W
#终止条件：1.迭代次数达到iter_num 2.所有样本分类正确
def PLA(X,Y,W0,iter_num):
    beginTime = datetime.datetime.now()
    W = W0.copy()
    correctFlag = True
    for i in range(iter_num):
        correctFlag = True
        for j in range(len(X)):
            if Sign(np.dot(X[j],W)) != Y[j]:
                W += Y[j]*X[j]
                correctFlag = False
        if correctFlag:
            print("PLA算法实际迭代次数：",i)
            break
    endTime = datetime.datetime.now()
    print("PLA算法运行时间：",endTime-beginTime)
    return W

#Pocket算法,输入样本集X，标签集Y，初始权重W0，迭代次数iter_num
#输出最终权重W
#终止条件：1.迭代次数达到iter_num 2.所有样本分类正确
def Pocket(X,Y,W0,iter_num):
    beginTime = datetime.datetime.now()
    W = W0.copy()
    W_pocket = W0.copy()
    W_error = 0
    W_pocket_error = 0
    correctFlag = True
    for i in range(iter_num):
        correctFlag = True
        for j in range(len(X)):
            if Sign(np.dot(X[j],W)) != Y[j]:
                W += Y[j]*X[j]
                for k in range(len(X)):
                    if Sign(np.dot(X[k],W)) != Y[k]:
                        W_error += 1
                    if Sign(np.dot(X[k],W_pocket)) != Y[k]:
                        W_pocket_error += 1
                if W_error < W_pocket_error:
                    W_pocket = W.copy()
                W_error = 0
                W_pocket_error = 0
                correctFlag = False
        if correctFlag:
            print("Pocket算法实际迭代次数：",i)
            break
    endTime = datetime.datetime.now()
    print("Pocket算法运行时间：",endTime-beginTime)
    return W_pocket

#可视化数据集,显示样本和分类面
#输入样本集X，标签集Y，权重W
#标签为1用圆圈表示，标签为-1用叉表示，分类面为W[0]+W[1]*x+W[2]*y=0表示的直线
def visualization(X,Y,W,type):
    plt.figure()
    for i in range(len(X)):
        if Y[i] == 1:
            plt.scatter(X[i][1],X[i][2],c='r',marker='o')
        else:
            plt.scatter(X[i][1],X[i][2],c='b',marker='x')
    x = np.linspace(-6,6,100)
    y = -(W[0]+W[1]*x)/W[2]
    plt.plot(x,y)
    if type == "PLA":
        plt.title("PLA")
    else:
        plt.title("Pocket")
    plt.show()
    
#主函数
if __name__ == '__main__':
    X1,Y1,X2,Y2 = GenData(data_size)
    X1 = Augment(X1)
    X2 = Augment(X2)
    #X = np.vstack((X1,X2))
    #Y = np.hstack((Y1,Y2))
    X_train = np.vstack((X1[:int(data_size*0.8)],X2[:int(data_size*0.8)]))
    Y_train = np.hstack((Y1[:int(data_size*0.8)],Y2[:int(data_size*0.8)]))
    X_test = np.vstack((X1[int(data_size*0.8):],X2[int(data_size*0.8):]))
    Y_test = np.hstack((Y1[int(data_size*0.8):],Y2[int(data_size*0.8):]))
    '''plt.figure()
    for i in range(len(X_train)):
        if Y_train[i] == 1:
            plt.scatter(X_train[i][1],X_train[i][2],c='r',marker='o')
        else:
            plt.scatter(X_train[i][1],X_train[i][2],c='b',marker='x')
    plt.show()
    #x = np.linspace(0,6,100)'''
   # print(X,Y)
   # X = Augment(X)
    #训练
    W_PLA = PLA(X_train,Y_train,W0_PLA,iter_num)
    W_Pocket = Pocket(X_train,Y_train,W0_Pocket,iter_num)
    print("训练结束")
    print("PLA算法最终权重：",W_PLA)
    print("Pocket算法最终权重：",W_Pocket)
    visualization(X_train,Y_train,W_PLA,"PLA")
    visualization(X_train,Y_train,W_Pocket,"Pocket")
    print("PLA算法训练集正确率：",Accuracy(X_train,Y_train,W_PLA))
    print("Pocket算法训练集正确率：",Accuracy(X_train,Y_train,W_Pocket))
    #测试
    visualization(X_test,Y_test,W_PLA,"PLA")
    visualization(X_test,Y_test,W_Pocket,"Pocket")
    print("PLA算法测试集正确率：",Accuracy(X_test,Y_test,W_PLA))
    print("Pocket算法测试集正确率：",Accuracy(X_test,Y_test,W_Pocket))
   
    
