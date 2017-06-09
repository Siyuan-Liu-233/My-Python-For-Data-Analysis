import numpy as np

data1=[6,7.5,8,0,1]
data2=[[1,2,3],[4,5,6]]
arr1=np.array(data1,dtype=np.float64)
arr2=np.array(data2,dtype=np.int32)
arr1.dtype
arr1.astype(np.int32)
#字符串全是数字 用astype转换为数值
arr.astype(np.int32)


np.zeros(10)
np.zeros((3,6))

np.arange(15)

np.random.randn(7,4) #生成正态分布数据

#and or 对Bool型 索引无效 得用 & | 
#Bool型索引会创建新副本

#切片 不复制数据

#花式索引
arr[[1,3,7]] #arr的1，1，3，7行
#选取方形区域
np.ix_
#花式索引是复制数据


#转置  返回源数据 不复制
arr.T
#叉乘 dot
#内积 np.dot(arr.T,arr)



#高维转置
arr=np.arange(16).reshape((2,2,4))
arr.transpose((1,0,2))
#高维交换轴
arr.swapaxes(1,2)
#也是返回源数据视图