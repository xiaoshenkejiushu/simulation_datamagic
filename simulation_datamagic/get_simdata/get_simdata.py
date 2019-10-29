# -*- coding: utf-8 -*-
'''
这个文件是用来根据原始的数据分布得到它的模拟数据分布的。
四个环节计算所得的均值分别为4.5、14.7、6、4.5
'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import copy
import math

#1.01这是做的各个环节的服务时间数据的读取和模拟
#分切录审+到达时间
dforig_classify = pd.read_csv('E:/code2019/simulation_datamagic/raw_data/classify_time.csv', encoding = 'utf-8')
dforig_cut = pd.read_csv('E:/code2019/simulation_datamagic/raw_data/cut_time.csv', encoding = 'utf-8')
dforig_input = pd.read_csv('E:/code2019/simulation_datamagic/raw_data/input_time.csv', encoding = 'utf-8')
dforig_review = pd.read_csv('E:/code2019/simulation_datamagic/raw_data/review_time.csv', encoding = 'utf-8')
dforig_applytime = pd.read_csv('E:/code2019/simulation_datamagic/raw_data/apply_time.csv',encoding = 'utf-8')#到达时间


def get_simulation_data(df_a,threshold,range_max,bins_count):
    #生成处理时间的部分
    df_a = df_a[df_a['action_time']>threshold]#二者数量不一致是因为这里设了一个门槛对其进行了筛选。
    
    df_a['action_time'] = df_a['action_time']
    print('--------这是真实的------------')
    print(df_a['action_time'].min())
    print(df_a['action_time'].mean())
    print(df_a['action_time'].std())#标准差
    print(df_a['action_time'].max())
    
    result = pd.DataFrame()
    exp_params = stats.expon.fit(np.array(df_a['action_time']))#loc指的是最小值，scale指的是均值减去最小值
    print('最小值和均值分别为',(exp_params[0],exp_params[1]))
    exp = stats.expon.rvs(exp_params[0],exp_params[1],4300)#通过最小值和均值即可生成模拟数列，exp_params[0],exp_params[1]
    result['sim_time'] = exp
    print('--------这是模拟的------------')
    print(result['sim_time'].min())
    print(result['sim_time'].mean())
    print(result['sim_time'].std())#标准差
    print(result['sim_time'].max())

#    bins_count = 50#直方图中柱状体的个数，#normed=True是频率图，默认是频数图
    plt.hist(np.array(df_a['action_time']),color='red',bins=bins_count,range = (0,range_max), alpha=0.9,label='Raw data') #绘制直方图, normed=True
    plt.hist(exp, bins=bins_count,color='blue',range = (0,range_max), alpha=0.5,label='Sim data') #绘制直方图., normed=True 
    plt.legend() # 显示图例
    plt.xlabel('value')
    plt.ylabel('quantity')
    plt.margins(0.02) 
    plt.show()
    result = result.sample(n=1200)
    return(result)

dfsim_classify = get_simulation_data(dforig_classify,1,50,25)
#dfsim_classify.to_csv("D:/code/pythonfile/simulation/data/output_data/sim_classify.csv",index =False)
#
dfsim_cut = get_simulation_data(dforig_cut,2,75,25)
#dfsim_cut.to_csv("D:/code/pythonfile/simulation/data/output_data/sim_cut.csv",index =False)
#
dfsim_input = get_simulation_data(dforig_input,1,50,25)

dforig_input['action_time'].plot(kind='kde',color='red',label='Raw data')
pd.Series(dfsim_input['sim_time']).plot(kind='kde',color='blue',label='Sim data')
plt.legend() # 显示图例
plt.xlabel('value')
plt.ylabel('density')
plt.show()
#dfsim_input.to_csv("D:/code/pythonfile/simulation/data/output_data/sim_input.csv",index =False)
#
dfsim_review = get_simulation_data(dforig_review,1,50,25)
#dfsim_review.to_csv("D:/code/pythonfile/simulation/data/output_data/sim_review.csv",index =False)


#1.02这是做的总处理时间的读取和模拟，暂时废弃了，我他妈的做错了

#dfonly_deal = pd.DataFrame()
#dfonly_deal['action_time'] = dforig_applytime['max_begin']
#dfsim_deal = get_simulation_data(dfonly_deal,100,1500,50)
#dfsim_deal.to_csv("D:/code/pythonfile/simulation/data/output_data/sim_deal.csv")



##1.03真实到达时间间隔的处理和模拟
#dfonly_apply = pd.DataFrame()
##这一段是生成了两个list，一个头加零，一个尾加零，然后做差，这样就产生了逐行相减的效果。
#apply_list = list(dforig_applytime['apply_time'])
#apply_list_head = copy.deepcopy(apply_list)
#apply_list_tail = copy.deepcopy(apply_list)
#apply_list_head.insert(0,0)
#apply_list_tail.append(0)
#apply_list = list(np.array(apply_list_tail)-np.array(apply_list_head))
#apply_list.remove(apply_list[0])
#
#
#def get_log(x):
#    logx = math.log(x)
#    return logx
###生成了一个新的dataframe，这个里面就是所有的时间间隔
#dfonly_apply['业务ID'] = dforig_applytime['业务ID']
#dfonly_apply['action_time'] = apply_list
#dfonly_apply = dfonly_apply[dfonly_apply['action_time']>0]
##dfonly_apply['action_time'] = dfonly_apply['action_time'].apply(get_log)#对数化
##dfsim_apply = get_simulation_data(dfonly_apply,4,10,15)#对数化
#dfsim_apply = get_simulation_data(dfonly_apply,0,600,15)







##0724新加，三选一式的生成
#c_t_l = [4,6,8]
#classify_time_list = []
#test = np.random.randint(3,size=1200)
#for i in range(len(test)):
#    classify_time_list.append(c_t_l[test[i]])


#1.04 虚假的到达间隔的模拟,最小值为0，时间间隔均值为12

dfsim_total = pd.DataFrame()
exp = stats.expon.rvs(0,3,1200)#通过最小值和均值即可生成模拟数列，exp_params[0],exp_params[1]
dfsim_total['arrive_time'] = exp
dfsim_total['classify_time'] = list(dfsim_classify['sim_time'])
#dfsim_total['classify_time'] = classify_time_list
dfsim_total['cut_time'] = list(dfsim_cut['sim_time'])
dfsim_total['input_time'] = list(dfsim_input['sim_time'])
dfsim_total['review_time'] = list(dfsim_review['sim_time'])

#0714 新加
rel_exp_sim = stats.norm.rvs(loc=80, scale=5, size=1200) #松弛时间服从均值为15的指数分布

dfsim_total['relaxation_time'] = rel_exp_sim


dfsim_total.to_csv("E:/code2019/simulation_datamagic/sim_0927_a3c4.5-14.7-6-4.5-r80_1.csv",index =False)

























