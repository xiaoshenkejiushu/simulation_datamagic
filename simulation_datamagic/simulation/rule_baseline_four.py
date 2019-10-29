# -*- coding: utf-8 -*-
'''
这个文件创作与0927，
做出四个环节联合后的单规则baseline

四个环节计算所得的均值分别为4.5、14.7、6、4.5
四个环节的总加工时长大约为30s


'''
import multiprocessing
import random
import simpy
import pandas as pd 
import numpy as np
from functools import partial, wraps
import copy
from rule_baseline_four_func import *

import matplotlib.pyplot as plt




global total_interval 
total_interval = 25#现用于reward生成，用这个数减去别的值，改用从表中读取。为了确定截止时间而设置，也就是截止时间减去开始时间
global NUM_BATCH
global BATCH_ORDERS
#导入变量
NUM_BATCH = 55
BATCH_ORDERS = 20
capacity_list = [2,4,2,4]


MIN_PATIENCE = 999999  # Min. customer patience
MAX_PATIENCE = 1000000  # Max. customer patience


RANDOM_SEED = 42#随机种子，确保结果的可复现性

#四个环节的处理时间的模拟
datapath = 'E:/code2019/simulation_datamagic/'
#得到模拟数据，到达间隔6、分类平均4.5，relaxation（超时时间限制）设置为10.5，下次设置时将每个环节的都标注出来。
df_total = pd.read_csv(datapath+'sim_0927_a3c4.5-14.7-6-4.5-r80_1.csv', encoding = 'utf-8')


#将数据变成想要的额那种格式，增加几列。
df_total = get_df_total(df_total)



g0_dict = {}
g0_dict['list_namec'] = pd.Series()
g0_dict['list_come'] = pd.Series()
g0_dict['list_begin'] = pd.Series()
g0_dict['list_wait'] = pd.Series()
g0_dict['list_finish']= pd.Series()
g0_dict['list_namef'] = pd.Series()

g1_dict = copy.deepcopy(g0_dict)
g2_dict = copy.deepcopy(g0_dict)
g3_dict = copy.deepcopy(g0_dict)

g0_dict['list_relax'] = pd.Series()
g0_dict['GENERATION'] = 'classify_time'
g1_dict['GENERATION'] = 'cut_time'
g2_dict['GENERATION'] = 'input_time'
g3_dict['GENERATION'] = 'cut_time'


sum_cut_number = 0




#目前的action是固定的，是用的spt
action_list = ['fifo','spt','srt']
ac_index = 2
action = action_list[ac_index]



#1  各个环节
# Setup and start the simulation
print('Bank renege')
random.seed(RANDOM_SEED)


#1.01  第0个机器，分类环节
env_g0 = simpy.Environment()
counter = simpy.PriorityResource(env_g0, capacity=capacity_list[0])

classify_data = []
monitor_cla = partial(monitor, classify_data)
patch_resource(counter, post=monitor_cla)

env_g0.process(source_cal(action,env_g0, counter,g0_dict,df_total,'classify'))
env_g0.run()




#1.02  第1个机器，切分环节
g1_dict['list_namec'] = g0_dict['list_namef']
g1_dict['list_come'] = g0_dict['list_finish']
env_g1 = simpy.Environment()
counter_hcq01 = simpy.PriorityResource(env_g1, capacity=capacity_list[1])

cut_data = []
monitor_cut = partial(monitor, cut_data)
patch_resource(counter_hcq01, post=monitor_cut)

env_g1.process(source_com(action,env_g1, counter_hcq01,g1_dict,df_total,'cut'))
env_g1.run()




#1.03  第2个机器，录入环节
g2_dict['list_namec'] = g1_dict['list_namef']
g2_dict['list_come'] = g1_dict['list_finish']

env_g2 = simpy.Environment()
counter_hcq02 = simpy.PriorityResource(env_g2, capacity=capacity_list[2])

input_data = []
monitor_input = partial(monitor, input_data)
patch_resource(counter_hcq02, post=monitor_input)

env_g2.process(source_com(action,env_g2, counter_hcq02,g2_dict,df_total,'input'))
env_g2.run()


#1.04  第3个机器，审核环节
g3_dict['list_namec'] = g2_dict['list_namef']
g3_dict['list_come'] = g2_dict['list_finish']

env_g3 = simpy.Environment()
counter_hcq03 = simpy.PriorityResource(env_g3, capacity=capacity_list[3])

review_data = []
monitor_review = partial(monitor, review_data)
patch_resource(counter_hcq03, post=monitor_review)

env_g3.process(source_com(action,env_g3, counter_hcq03,g3_dict,df_total,'review'))
env_g3.run()


##2  生成环节数据
get_huanjie_data(classify_data,cut_data,input_data,review_data,capacity_list)

#3   生成文档数据:每个order的到达、等待、开始、结束。
df_document_cl,df_document_cu,df_document_in,df_document_re = get_order_data(g0_dict,g1_dict,g2_dict,g3_dict,action)

#4  绘制最终结果
result_analyse(action,df_document_cl,df_document_cu,df_document_in,df_document_re)




