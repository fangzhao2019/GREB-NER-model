from openpyxl import load_workbook
import os
import numpy as np
from xpinyin import Pinyin
import newIndexCreate
#import relationCalculatingUndirected
import relationCalculatingDirected
import pagerank
p = Pinyin()
pg = pagerank.CPageRank()
#############################################相关参数#############################
alpha=0.85#pagerank算法中页面跳转的概率
beta=0.8#衡量车型大小相似度和价格相似度

print('......正在构建索引库......')
carIndexSet=newIndexCreate.indexCreate_main('汽车知识库/newCar.xlsx')
brandIndexSet=newIndexCreate.indexCreate_main('汽车知识库/newBrand.xlsx')
print('索引库构建完成\n')

print('......有向关系计算中......')
brandSet,brandWeight=relationCalculatingDirected.brandDataLoading()
brandCarSet,carSet,configSet,carWeight=relationCalculatingDirected.carDataLoading()
sizeRelation,priceRelation=relationCalculatingDirected.C2CrelationMat(carSet,configSet)
B2Crelation=relationCalculatingDirected.B2CrelationMat(brandSet,brandCarSet,carSet)
B2Brelation1,B2Brelation2=relationCalculatingDirected.B2BrelationMat(brandSet,carSet,brandCarSet,sizeRelation,priceRelation,carWeight)
print('关系计算完成\n')

def commentLoading(file):
    comment=[]
    with open(file,encoding='utf-8') as f:
        for row in f.readlines():
            if len(row)>0:
                comment.append(row.split('\t')[0])
    return comment

def candidateEntityExtract(comment,carIndexSet,brandIndexSet):
    commentIndexSet=[]
    length=len(comment)
    #对于每个字符，往后逐渐合并字符，判断合并字符或者其拼音是否有索引实体
    for i in range(len(comment)):
        carIndex=[]
        brandIndex=[]
        for j in range(max(0,i-10),i+1):
            for k in range(i,min(length,i+10)):
                word=''.join(comment[j:k+1])
                pinyin=newIndexCreate.word2pinyin(word)
                if word in carIndexSet.keys():
                    carIndex+=carIndexSet[word]
                if pinyin in carIndexSet.keys():
                    carIndex+=carIndexSet[pinyin]
                if word in brandIndexSet.keys():
                    brandIndex+=brandIndexSet[word]
                if pinyin in brandIndexSet.keys():
                    brandIndex+=brandIndexSet[pinyin]
        commentIndexSet.append([list(set(carIndex)),list(set(brandIndex))])
    return commentIndexSet

def entityCount(commentIndexSet):
    candidateCar={}
    candidateBrand={}
    for com in commentIndexSet:
        for car in com[0]:
            if not car in candidateCar.keys():
                candidateCar[car]=0
            candidateCar[car]+=1
        for brand in com[1]:
            if not brand in candidateBrand.keys():
                candidateBrand[brand]=0
            candidateBrand[brand]+=1
    candidateCar=list(candidateCar.keys())
    candidateBrand=list(candidateBrand.keys())
    return candidateCar,candidateBrand

def relationMatGenerating(candidateCar,candidateBrand,beta):
    length1=len(candidateBrand)
    length2=len(candidateCar)
    length=length1+length2
    relationMat=np.zeros((length,length))
    init_value=np.ones(length)
    for i in range(length):
        for j in range(length):
            if i==j:continue
            if i<length1 and j<length1:#品牌-品牌
                brand1,brand2=(candidateBrand[i],candidateBrand[j])
                relation1,relation2=relationCalculatingDirected.getB2Brelation_cut(brand2,brand1,brandSet,B2Brelation1,B2Brelation2)
                relationMat[i,j]=beta*relation1+(1-beta)*relation2
            if i<length1 and j>=length1:#车型-品牌
                brand1,car1=(candidateBrand[i],candidateCar[j-length1])
                relation=relationCalculatingDirected.getB2Crelation(brand1,car1,brandSet,carSet,B2Crelation)
                relationMat[i,j]=relation
            if i>=length1 and j>=length1:#车型-车型
                car1,car2=(candidateCar[i-length1],candidateCar[j-length1])
                relation1,relation2=relationCalculatingDirected.getC2Crelation_cut(car2,car1,carSet,sizeRelation,priceRelation)
                relationMat[i,j]=beta*relation1+(1-beta)*relation2
    for i in range(length):
        if i<length1:
            init_value[i]=relationCalculatingDirected.getBrandWeight(candidateBrand[i],brandSet,brandWeight)
        else:
            init_value[i]=relationCalculatingDirected.getCarWeight(candidateCar[i-length1],carSet,carWeight)
    init_value=init_value/sum(init_value)
    return init_value,relationMat

def candidateEntityMerge(candidateCar,candidateBrand,final_value):
    candidateEntityDic={}
    candidateEntity=candidateBrand+candidateCar
    for i in range(len(final_value)):
        candidateEntityDic[candidateEntity[i]]=final_value[i]
    return candidateEntityDic
            
def candidateEntityRanking(commentIndexSet,candidateEntityDic):
    sortedIndexSet=[]
    for i in range(len(commentIndexSet)):
        com=commentIndexSet[i]
        entitySet=[]
        for c in com[0]+com[1]:
            entitySet.append((c,str(candidateEntityDic[c])))
        sortedIndex=sorted(entitySet,key=lambda k:k[1],reverse=True)
        sortedIndexSet.append(sortedIndex)
    return sortedIndexSet

def resultSaving(filename,sortedIndexSet):
    with open(filename,'w',encoding='utf-8') as f:
        for data in sortedIndexSet:
            if len(data)>0:
                f.write('\t'.join([','.join(d) for d in data]))
            else:
                f.write('Null')
            f.write('\n')

def single_process(filename1,filename2):
    comment=commentLoading(filename1)
    commentIndexSet=candidateEntityExtract(comment,carIndexSet,brandIndexSet)
    candidateCar,candidateBrand=entityCount(commentIndexSet)
    init_value,relationMat=relationMatGenerating(candidateCar,candidateBrand,beta)
    final_value=pg.GetPR(relationMat.T, init_value, alpha, max_itrs=100, min_delta=0.0001)#是否需要转置
    candidateEntityDic=candidateEntityMerge(candidateCar,candidateBrand,final_value)
    sortedIndexSet=candidateEntityRanking(commentIndexSet,candidateEntityDic)
    resultSaving(filename2,sortedIndexSet)
    #for i in range(len(commentIndexSet)):
    #    print(comment[i],sortedIndexSet[i])

def batch_process(path,data_type):
    filepath='%s/%s'%(path,data_type)
    resultpath='%s/%sKnowledge'%(path,data_type)
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
    i=1
    for filename in os.listdir(filepath):
        if i%10==0:print(i)
        print('正在处理文件%s'%filename)
        datafile='%s/%s'%(filepath,filename)
        resultfile='%s/%s'%(resultpath,filename)
        single_process(datafile,resultfile)
        i+=1

path1='G:/论文实验——实体识别与链接/汽车命名实体识别/data/example.txt'
path2='G:/论文实验——实体识别与链接/汽车命名实体识别/data/exampleKnowledge.txt'
single_process(path1,path2)
#batch_process('G:/论文实验——实体识别与链接/汽车命名实体识别/data','trainData')
#batch_process('G:/论文实验——实体识别与链接/汽车命名实体识别/data','devData')
#batch_process('G:/论文实验——实体识别与链接/汽车命名实体识别/data','testData')

