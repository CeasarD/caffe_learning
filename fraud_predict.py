#-*- coding:utf-8 -*-
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ch
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,recall_score,classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import itertools
ch.set_ch()

def readFile(_filename):
    '''
数据读取函数
'''
    filetype=_filename.split('.')[-1]
    with open(_filename) as f:
        pd.set_option('max_row',None)
        if filetype=='txt':
            data=pd.read_table(f,header=None,encoding='gb2312',
                               delim_whitespace=True)
        if filetype=='csv':
            data=pd.read_csv(f)
    return data

##-----------------------------------------------------------------------
def plotFileData_h(_data,_class=-1):
    '''
源数据直方图,默认最后一列为类别
'''
    count_classes=pd.value_counts(_data[_data.columns[_class]],
                                  sort=True).sort_index()#求各个类别的数量并按照索引进行排序
    count_classes.plot(kind='bar')
    plt.title('Fraud class histogram')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()
    
##----------------------------------------------------------------------
def filedataStandardScaler(_data,_columnsNum):
    '''
源数据某列数据的标准化
_data:源数据
_columnsNum:需要标准化的列编号列表
'''
    columnsNames=[]
    for i in _columnsNum:
        columnsName=_data.columns[i]#需要标准化的列名
        columnsNames.append(columnsName)
        newName='norm'+columnsName#标准化后的列名
        normColumnData=StandardScaler().fit_transform(X=np.array(_data).T[i].reshape(-1,1))#标准化
        _data[newName]=normColumnData#将标准化的列加入数据中
    data=_data.drop(columnsNames,axis=1)#删除原有未标准化的列
    return data

##----------------------------------------------------------------------
def downSampling(_data):
    '''
对源数据中数量较多的类别进行随机采样使其与较低的类别的样本量保持一致
'''
    lessSampleNum=pd.value_counts(_data['Class'],sort=True).sort_index().iloc[-1]#取最后一个
    lessSampleIndex=np.array(_data[_data.Class==1].index)#取出class=1的样本索引
    moreSampleIndex=np.array(_data[_data.Class==0].index)#取出class=0的样本索引
    rdMoreSample=np.array(np.random.choice(moreSampleIndex,lessSampleNum,
                                           replace=False))#在所有class=1（较多的类）中随机取样，数量和较少的一致
    underSampleIndex=np.concatenate([lessSampleIndex,rdMoreSample])#下采样样本索引
    underSampleData=_data.iloc[underSampleIndex,:]#下采样数据
    underSampleValue,underSampleLabel=valueLabelSep(underSampleData)#下采样的数据和标签
    print("Percentage of normal transactions: ",
          len(underSampleData[underSampleData.Class == 0])/len(underSampleData))
    print("Percentage of fraud transactions: ",
          len(underSampleData[underSampleData.Class == 1])/len(underSampleData))
    print("Total number of transactions in resampled data: ",
          len(underSampleData))
    return underSampleValue,underSampleLabel
    

##----------------------------------------------------------------------
def upSampling(_data):
    '''
    过采样
'''
    
    features,labels=valueLabelSep(_data)
    #xTrain,xTest,yTrain,yTest=ms.train_test_split(features,labels,test_size=0.2,random_state=0)
    overSampler=SMOTE(random_state=0)
    os_xTrain,os_yTrain=overSampler.fit_sample(features,labels)
    print(len(os_yTrain[os_yTrain==1]))
    print(len(os_yTrain[os_yTrain==0]))
    os_xTrain=pd.DataFrame(os_xTrain)
    os_yTrain=pd.DataFrame(os_yTrain)
    return os_xTrain,os_yTrain

##---------------------------------------------------------------------
def valueLabelSep(_data):
    '''
将样本值和标签进行分割
'''
    value=_data.iloc[:,data.columns!='Class']
    label=_data.iloc[:,data.columns=='Class']
    return value,label
##----------------------------------------------------------------------
def trainTestSep(_dataX,_dataY,_testSize=0.3):
    '''
将数据集按比例分成训练集和测试集
_testSize：测试集所占比例
'''
    xTrain,xTest,yTrain,yTest=train_test_split(_dataX,_dataY,
                                               test_size=_testSize,random_state=0)
    print("Number transactions train dataset: ", len(xTrain))
    print("Number transactions test dataset: ", len(xTest))
    print("Total number of transactions: ", len(xTrain)+len(xTest)) 
    return xTrain,xTest,yTrain,yTest
##-----------------------------------------------------------------------
def lrKfoldRcScores(_xTrain,_yTrain,_cParam=[0.01,0.1,1,10,100]):
    '''
训练集交叉验证函数，输出在不同惩罚参数下，逻辑回归的召回率
_cParam:惩罚率
'''
    fold=KFold(len(_yTrain),5,shuffle=False)#划分五组，每组一个训练集一个验证集
    cParam=_cParam#惩罚率
    resultsTable=pd.DataFrame(index=range(len(cParam),2),
                              columns=['C_parameter','Mean Recall Score'])#建立不同惩罚率的召回率矩阵
    resultsTable['C_parameter']=cParam
    j=0
    for c_param in cParam:
        print('---------------------------------')
        print('C parameter:',c_param)
        print('---------------------------------')
        print('')
        recallAccs=[]
        for iteration,indices in enumerate(fold,start=1):
            lr=LogisticRegression(C=c_param,penalty='l1')#设置逻辑回归模型
            lr.fit(_xTrain.iloc[indices[0],:],_yTrain.iloc[indices[0],:].values.ravel())#使用逻辑回归模型进行训练
            yPred=lr.predict(_xTrain.iloc[indices[1],:].values)#使用模型进行预测
            recallAcc=recall_score(_yTrain.iloc[indices[1],:].values,yPred)#计算召回率
            recallAccs.append(recallAcc)
            print('Iteration ',iteration,':recall score= ',recallAcc)
        resultsTable.loc[j,'Mean Recall Score']=np.mean(recallAccs)#当前惩罚率下的平均召回率
        j+=1
        print('')
        print('Mean recall score ',np.mean(recallAccs))
        print('')
    bestC=resultsTable.loc[resultsTable['Mean Recall Score'].idxmax()]['C_parameter']#最佳惩罚率
    print('********************************************')
    print('Best model to choose from cross validation is with C parameter= ',bestC)
    print('********************************************')
    
    return bestC,resultsTable
##-----------------------------------------------------------------------
def lrTest(_xTrain,_xTest,_yTrain,_yTest,_bestC,_penalty='l1'):
    '''
使用测试集

'''
    lr=LogisticRegression(C=_bestC,penalty=_penalty)#设置逻辑回归模型
    lr.fit(_xTrain,_yTrain.values.ravel())#使用逻辑回归模型进行训练
    yPred=lr.predict(_xTest.values)#使用模型进行预测
    recallAcc=recall_score(_yTest.values,yPred)#计算召回率
    return recallAcc,lr

def lrPredict(_xtest,_lr):
    '''
预测分类
'''
    yPred=_lr.predict(np.array(_xtest.values).reshape(1,-1))
    return yPred

def confusionMatrix(_xTest,_yTest,_lr):
    '''
生成混合矩阵
_lr:训练好的逻辑回归模型
'''
    ypred=_lr.predict(_xTest.values)
    cnfMatrix=confusion_matrix(_yTest,ypred)
    
    print("Recall metrix in the testing dataset:",cnfMatrix[1,1]/(cnfMatrix[1,0]+cnfMatrix[1,1]))
    return cnfMatrix

def plotConfusionMatrix(_cm,_classes,_title='Confusion matrix',_cmap=plt.cm.Blues):
    '''
绘制混淆矩阵
cm：混淆矩阵
classes：类别名称
cmap：颜色
'''
    plt.imshow(_cm,interpolation='nearest',cmap=_cmap)
    plt.title(_title)
    plt.colorbar()
    tickMarks=np.arange(len(_classes))
    plt.xticks(tickMarks,_classes,rotation=0)
    plt.yticks(tickMarks,_classes)
    thresh=_cm.max()/2
    for i, j in itertools.product(range(_cm.shape[0]), range(_cm.shape[1])):
        plt.text(j, i, _cm[i, j],
                 horizontalalignment="center",
                 color="white" if _cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
    






##----------------------------------------------------------------------
##data=readFile('C:\\Users\\Ceasar\\Downloads\\horseColicTraining.txt')
data=readFile('horseColicTraining.txt')
print(data.head())
data.columns=[str(e) for e in np.arange(22)]
##data.rename(columns={'21':'Class'},inplace=True)
data=filedataStandardScaler(data,[e for e in np.arange(21)])
data['Class']=data['21'].values
data=data.drop(['21'],axis=1)
print(data.head())
plotFileData_h(data)
####data1=filedataStandardScaler(data,[0,29])
####value,label=downSampling(data)
##value,label=upSampling(data)
####value,label=valueLabelSep(data)
##xtrain,xtest,ytrain,ytest=trainTestSep(value,label)
####print(xtrain.shape)
####print(ytrain.shape)
##c,res=lrKfoldRcScores(xtrain,ytrain)
####print(res)
##rec,lr=lrTest(xtrain,xtest,ytrain,ytest,c)
##print(rec)
####print(lr)
####pred_y=lrPredict(data.iloc[1,data.columns!='Class'],lr)
####print(pred_y)
####print(data.iloc[1,data.columns=='Class'])
##class_name=[0,1]
##cnf_matrix=confusionMatrix(xtest,ytest,lr)
##np.set_printoptions(precision=2)
##plt.figure()
##plotConfusionMatrix(cnf_matrix,_classes=class_name)
plt.show()


