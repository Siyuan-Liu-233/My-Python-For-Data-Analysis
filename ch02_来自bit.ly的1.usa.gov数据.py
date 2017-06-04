import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


#得到记录
path="data/ch02/usagov_bitly_data2012-03-16-1331923249.txt"
records=[json.loads(i) for i in open(path)]

#得到时区数据 并计数

#方法1 运用collections 的 Count
#time_zone=[i["tz"] for i in records if "tz" in i]
#time_zone=Counter(time_zone).most_common()

#方法2 运用pandas
frame=pd.DataFrame(records)
tz_fill=frame["tz"].fillna("Missing")
tz_fill[tz_fill=='']='Unknow'
tz_counts=tz_fill.value_counts()
plt.figure(1)
tz_counts[:10].plot(kind='barh')  #对前十个数据作条形图
plt.savefig("ch02_1")


#统计agent
agent=pd.Series([i.split()[0] for i in frame.a.dropna()])
agent_counts=agent.value_counts()
plt.figure(2)
agent_counts[:10].plot(kind='barh',figsize=(20,6))  #对前十个数据作条形图
plt.savefig("ch02_2")


#假设agent含有Windows为Windows用户 对Windows 用户进行统计
wframe=frame[frame.a.notnull()]
o_syetem=np.where(wframe.a.str.contains('Windows'),\
    'Windows','Not Windows')
tz_os=wframe.groupby(['tz',o_syetem]).size().unstack().fillna(0)
sort_tz_os=tz_os.sum(axis=1).argsort()
tz_os=tz_os.take(sort_tz_os)[-10:]
tz_os_norm=tz_os.div(tz_os.sum(axis=0),axis=1)

tz_os_norm.plot(kind='barh',stacked=True,figsize=(20,6))
plt.savefig("ch02_3")
plt.show()
