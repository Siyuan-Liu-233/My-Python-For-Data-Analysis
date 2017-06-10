import numpy as np

data1=[6,7.5,8,0,1]
data2=[[1,2,3],[4,5,6]]
arr1=np.array(data1,dtype=np.float64)
arr2=np.array(data2,dtype=np.int32)
arr1.dtype
arr1.astype(np.int32)
#字符串全是数字 用astype转换为数值
# arr=np.array([1,2,3])
# arr=arr.astype(np.str)
# print(arr)


np.zeros(10)
np.zeros((3,6))

np.arange(15)

np.random.randn(7,4) #生成正态分布数据

#and or 对Bool型 索引无效 得用 & | 
#Bool型索引会创建新副本

#切片 不复制数据

#花式索引
#arr[[1,3,7]] #arr的1，1，3，7行
#选取方形区域
#np.ix_
#花式索引是复制数据


#转置  返回源数据 不复制
#arr.T
#叉乘 dot
#内积 np.dot(arr.T,arr)



#高维转置
#arr=np.arange(16).reshape((2,2,4))
#arr.transpose((1,0,2))
#高维交换轴
#arr.swapaxes(1,2)
#也是返回源数据视图

# arr=np.arange(10)
# np.sqrt(arr)
# print(np.sqrt(arr))

# np.exp(arr)
# x=np.random.random(8)
# y=np.random.random(8)
# print(x,y)
# print(np.maximum(x,y))

# arr=np.random.random(7)*5
# print(arr)
# print(np.modf(arr))

#利用数组进行数据处理
# points=np.arange(-5,5,0.01)
# xs,ys=np.meshgrid(points,points)
# print(xs,ys)
# z=np.sqrt(xs**2+ys**2)
# print(z)
# import matplotlib.pyplot as plt
# plt.imshow(z,cmap=plt.cm.gray)
# plt.colorbar()
# plt.show()

# #将条件逻辑表述为数组运算
# xarr=np.arange(1.1,1.6,0.1)
# yarr=np.arange(2.1,2.6,0.1)
# cond=np.array([True,False,True,True,False])
# result=[(x if c else y) for x,y,c in zip(xarr,yarr,cond)]
# print(result)

# #np.where(a,b,c) a为判断 真 则 把那个位置换成b 否则换成c
# arr=np.random.randn(4,4)	#标准正态分布随机
# m=np.where(cond,xarr,yarr)
# print(m)

# print(arr)
# m=np.where(arr>0,2,-2) #把arr中大于0的换成2 否则换成-2
# print(m)


#数学和统计方法 
#sum mean std var min max argmin argmax cumsum cumprod
# arr=np.random.randn(5,4)
# print(arr)
# print(arr.sum(1))
# print(arr.mean()) 
# print(arr.mean(axis=1)) #axis=1 表示求每一行的平均值

# print(arr.mean(1))
# print(np.mean(arr[0,:]))

# #用于布尔型数组
# arr=np.random.randn(100)
# print((arr>0).sum()) 	#正数数量

# bools=np.array([False,False,True,True,False])
# print(bools.any()) 		#判断是否存在True
# print(bools.all())		#判断是否全为False

# #排序
# arr=np.random.randn(8)
# print(arr)
# arr.sort()
# print(arr)
# arr=np.random.randn(5,3)
# print(arr)
# arr.sort(1)	#对横轴升序
# print(arr)

# large_arr=np.random.randn(1000)
# large_arr.sort()
# x=large_arr[int(0.05*len(large_arr))] 	#第5%大的数
# print(x)


#其他集合逻辑
# names=np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
# print(np.unique(names))   #返回存在的值

# value=np.array([6,0,0,3,2,5,6])
# print(np.in1d(value,[2,3,6]))		#判断value中每个值是否在[2,3,6]中

# x=[1,2,3,7,7,7,2,3]
# y=[1,7,7,3,4,5,5,5]
# print(np.intersect1d(x,y)) 		#x,y中共同元素
# print(np.union1d(x,y))  		#x,y并集
# print(np.setdiff1d(x,y)) 		#在x中不在y中
# print(np.setxor1d(x,y)) 		#仅存在于一个数组中

# #文件输入输出
# arr=np.arange(10)
# np.save('some_array',arr)	#会自动加上.npy的后缀
# out=np.load('./some_array.npy')
# print(out)

# arr2=np.arange(12).reshape(3,4)
# np.savez('array2',a=arr,b=arr2)  #多个数组保存
# out=np.load('array2.npz')
# print(out['b'])

# arr=np.loadtxt('array.txt',delimiter=',')
# print(arr)

# #线性代数
# x=np.array([[1,2,3],[4,5,6]])
# y=np.arange(6).reshape(3,2)
# print(x,y)
# print(x.dot(y))
# print(np.dot(x,y))

# from numpy.linalg import inv,qr
# x=np.random.randn(4,4)
# print(inv(x))
# x=np.matrix(x)
# print(x.I)

# #numpy.random函数 随机数
# #seed 随机数种子
# #shuffle 随机排列
# a=np.array([[1,2,3,4,5,6,7,8],
# 	[2,3,4,5,6,7,8,9]])
# np.random.shuffle(a[1])
# print(a)
# #rand 产生均匀分布样本值
# #randint 从给定上下限选整数
# print(np.random.randint(1,6)) #不包含6
# #randn 产生正态分布 平均0 标准差1 
# #binomial 产生二项分布
# #normal 产生正态分布
# #uniform 产生[0，1)均匀分布