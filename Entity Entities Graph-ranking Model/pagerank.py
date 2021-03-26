import numpy as np
 
class CPageRank(object):
    '''实现PageRank Alogrithm
    '''
    def __init__(self):
        self.PR = [] #PageRank值
 
    def GetPR(self, IOS, init_PR, alpha, max_itrs, min_delta):
        '''幂迭代方法求PR值
        :param IOS       表示网页出链入链关系的矩阵,是一个左出链矩阵
        :param alpha     阻尼系数α，一般alpha取值0.85
        :param max_itrs  最大迭代次数
        :param min_delta 停止迭代的阈值
        '''
        #归一化转移矩阵和初始节点值
        self.PR=init_PR/sum(init_PR)
        #print('初PR值表:', self.PR)
        helpS=IOS/(IOS.sum(axis=0)+0.000001)

        #IOS左出链矩阵, a阻尼系数alpha, N网页总数
        N = np.shape(IOS)[0]
        if N==0:
            return np.array([])
        #所有分量都为1的列向量
        e = np.ones(shape=(N, 1))
        #计算网页出链个数统计
        L = [np.count_nonzero(e) for e in IOS.T]
        
        #P[n+1] = AP[n]中的矩阵A
        A = alpha*helpS + ((1-alpha)/N)*np.dot(e, e.T)
        #print('左出链矩阵:\n', IOS)
        #print('左PR值贡献概率矩阵:\n', helpS)
        #幂迭代法求PR值
        for i in range(max_itrs):
            #使用PR[n+1] = APR[n]递推公式，求PR[n+1]
            old_PR = self.PR
            self.PR = np.dot(A, self.PR)
            #如果所有网页PR值的前后误差 都小于 自定义的误差阈值，则停止迭代
            D = np.array([old-new for old,new in zip(old_PR, self.PR)])
            ret = [e < min_delta for  e in D]
            if ret.count(True) == N:
                #print('迭代次数:%d, succeed PR:\n'%(i+1), self.PR)
                break
        return self.PR
    
if __name__=='__main__':
    IOS = np.array([[0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0]], dtype=float)
    
    init_PR=np.array([3,2,5,3,6])

    pg = CPageRank()
    ret = pg.GetPR(IOS, init_PR, alpha=0.85, max_itrs=100, min_delta=0.0001)
    print('最终的PR值:\n', ret)
