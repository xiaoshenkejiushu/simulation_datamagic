# -*- coding: utf-8 -*-

"""
肖申克记录于927
将联合起来的四个环节的调度用单规则调度试一试，看看结果。

这个文件主要是调度的函数
"""
import simpy
import pandas as pd 
import numpy as np
from functools import partial, wraps
import random
import copy
import matplotlib.pyplot as plt


global total_interval 
total_interval = 25#现用于reward生成，用这个数减去别的值，改用从表中读取。为了确定截止时间而设置，也就是截止时间减去开始时间
global NUM_BATCH
global BATCH_ORDERS

NUM_BATCH = 55
BATCH_ORDERS = 20
capacity_list = [2,4,2,4]

#导入变量


MIN_PATIENCE = 999999  # Min. customer patience
MAX_PATIENCE = 1000000  # Max. customer patience


#0402添加，这两个函数是用来看这个环节有多少人的。
def patch_resource(resource, pre=None, post=None):
    
    def get_wrapper(func):
        
        # Generate a wrapper for put/get/request/release
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is the actual wrapper
            # Call "pre" callback
            if pre:
                pre(resource)
            # Perform actual operation
            ret = func(*args, **kwargs)
            # Call "post" callback
            if post:
                post(resource)
            return ret
        return wrapper
    # Replace the original operations with our wrapper
    for name in ['put', 'get', 'request', 'release']:
        if hasattr(resource, name):
            setattr(resource, name, get_wrapper(getattr(resource, name)))       
def monitor(data, resource):
    item = [
            resource._env.now,  # The current simulation time
         resource.count,  # The number of users
        len(resource.queue)] # The number of queued processes)
    
    data.append(item)
    
    
#927新加,将数据变成想要的额那种格式，增加几列。
def get_df_total(df_total):
    arrmom_list = [0]
    for j in range(len(df_total['arrive_time'])-1):
        arr_mom = sum(df_total['arrive_time'][:j+1])
        arrmom_list.append(arr_mom)
    df_total['arrive_moment'] = arrmom_list   
    
    
    #0819新加数据
    df_total['due'] = df_total['arrive_moment'] + df_total['relaxation_time']
    df_total['dynamic'] = df_total['relaxation_time'] - df_total['classify_time']- df_total['cut_time']- df_total['input_time']- df_total['review_time']
    return df_total


#fifo先进先出,spt最短加工时间,srt最短松弛时间
def schduel_prio(action,num,df_simtime,huanjie_name):#得到优先级，实现三种调度方式的切换
    if action == 'fifo':
        prio = 1
    elif action == 'lifo':#都进行归一化
        prio = np.mean(df_simtime['arrive_moment'])/(df_simtime['arrive_moment'][num]+10)
    elif action == 'spt':#都进行归一化
        prio = df_simtime[huanjie_name+'_time'][num]/np.mean(df_simtime[huanjie_name+'_time'])
    elif action == 'srt':
        prio = df_simtime['relaxation_time'][num]/np.mean(df_simtime['relaxation_time'])#需要进行随机生成
    elif action == 'ed':
        prio = df_simtime['due'][num]/np.mean(df_simtime['due'])
    elif action == 'dsrt': 
        prio = df_simtime['dynamic'][num]/np.mean(df_simtime['dynamic'])
    else:
        print('输入异常')
        prio = 0.5
    
    return prio


def source_cal(action,env,counter,gen_dict,df_simtime,huanjie_name):
    huanjie_name = 'classify'
    sum_doc_num = 0
    arrive_time = 0
    
    
    for i in range(NUM_BATCH):

        #各个批次的调度
        for j in range(sum_doc_num,sum_doc_num+BATCH_ORDERS):
            serve_time = df_simtime['classify_time'][j]#得到模拟数据
            prio = schduel_prio(action,j,df_simtime,huanjie_name)#优先级，这也是调度的根本所在。
            order_name = 'doc%02d' % j#订单名称，用doc为了方便ui显示，后面有个3也要改。
            c = document(prio,env,order_name,gen_dict,counter, serve_time)#根据函数生成一个doc
            env.process(c)
            t = df_simtime['arrive_time'][j]#到达间隔从列表中读取
#            t = random.expovariate(lamda)#废弃，这是直接从本文件中生成随机数。到达时间服从指数分布,此处的t为间隔时间,1/10为lamda,均值为10
            gen_dict['list_come'][order_name] = arrive_time#到达时间list
            gen_dict['list_namec'][order_name] = order_name#名字
            gen_dict['list_relax'][order_name] = df_simtime['relaxation_time'][j]#relaxa 时间
            arrive_time += t 
            sum_doc_num += 1
            yield env.timeout(t)


#先把一变多去掉了，试试正常环境的。     
def source_com(action,env,counter,gen_dict,df_simtime,huanjie_name):   
    """Source generates Doc randomly"""
    sum_doc_num = 0    

    for i in range(NUM_BATCH):
        
        #各个批次的调度
        for j in range(sum_doc_num,sum_doc_num+BATCH_ORDERS):#从上一个环节的结束时间，得到本环节的到达时间
            if j == 0:
                t = gen_dict['list_come'][j]#到达时间服从指数分布,此处的t为间隔时间
            else:
                t = gen_dict['list_come'][j] - gen_dict['list_come'][j-1]
            sum_doc_num += 1
            yield env.timeout(t)
            
            order_name = gen_dict['list_namec'][j]#name和come都可以直接用j，但是从文件中读过来的没法直接用j。
            col_name = gen_dict['GENERATION']#0817新加
            ser_index = int(order_name[3:])#0817新加，根据名字得到index，方便从文件中读取相关时间信息
            serve_time = df_simtime[col_name][ser_index]#得到模拟数据
            prio = schduel_prio(action,ser_index,df_simtime,huanjie_name)#优先级，这也是调度的根本所在。
            c = document(prio,env,order_name,gen_dict,counter, serve_time)
            env.process(c)
            
        
def document(prio,env, name, gen_dict,counter,serve_time):
    """Doc arrives, is served and leaves."""
    generation = gen_dict['GENERATION']
    arrive = env.now
    print('%7.4f %s %s: Here I am' % (arrive,generation, name))
    with counter.request(priority=prio) as req:
        patience = random.uniform(MIN_PATIENCE, MAX_PATIENCE)#忍耐时间服从均匀分布        
        # Wait for the counter or abort at the end of our tether
        results = yield req | env.timeout(patience)#这句话挺关键的但是我目前没有看懂它是什么意思
        begin_time = env.now
        gen_dict['list_begin'][name] =begin_time#添加开始服务时间
        wait = env.now - arrive
        gen_dict['list_wait'][name] =wait#添加等待时间
        if req in results:
            # We got to the counter
            print('%7.4f %s%s: Waited %6.3f' % (env.now,generation, name, wait))
            print('%7.4f %s%s:begin at %6.3f' % (env.now,generation, name, env.now))
            yield env.timeout(serve_time)
            print('%7.4f %s %s: Finished' % (env.now,generation, name))
            finished_time = env.now
            gen_dict['list_finish'][name] = finished_time#添加结束时间
            gen_dict['list_namef'][name] = name

        else:
            # We reneged
            print('%7.4f %s: RENEGED after %6.3f' % (env.now, name, begin_time))
            
            
            
def complete_df(df_huanjie,capacity_huanjie):#补全环节数据
    append_list = []#所需补全的数据
    #首先是数据补全操作
    for i in range(len(df_huanjie)):#得到需要补全的数据
        row_huanjie = df_huanjie.iloc[i]
        if row_huanjie['users']<capacity_huanjie:
            if row_huanjie['queue'] > 0:
                temp_time = row_huanjie['time']
                temp_users = row_huanjie['users'] +1
                temp_queue = row_huanjie['queue'] -1
                temp_list = [i,temp_time,temp_users,temp_queue]
                append_list.append(temp_list)
    
    for j in range(len(append_list)):#将行插入到dataframe中
        values=[append_list[j][1],append_list[j][2],append_list[j][3]]
        df_huanjie = pd.DataFrame(np.insert(df_huanjie.values, append_list[j][0]+1+j, values, axis=0))
    df_huanjie.columns = ['time','users','queue']
    return df_huanjie



#3   生成文档数据:每个order的到达、等待、开始、结束。            
def get_order_data(g0_dict,g1_dict,g2_dict,g3_dict,action):
    
    df_document_cl = pd.DataFrame()    
    df_document_cu = pd.DataFrame()
    df_document_in = pd.DataFrame()
    df_document_re = pd.DataFrame()
    
    
    df_document_cl['name'] = g0_dict['list_namec']
    df_document_cl['come'] = g0_dict['list_come'] 
    df_document_cl['wait'] = g0_dict['list_wait']  
    df_document_cl['begin'] = g0_dict['list_begin'] 
    df_document_cl['finish'] = g0_dict['list_finish']  
    df_document_cl['t_relax_time'] = g0_dict['list_relax'] 
    df_document_cl.to_csv('../result_data/'+'df_document_classify.csv', index = False)
    
    
    

    
    df_document_cu['name'] = g1_dict['list_namec']
    df_document_cu['come'] = g1_dict['list_come'] 
    df_document_cu['wait'] = g1_dict['list_wait'] 
    df_document_cu['begin'] = g1_dict['list_begin'] 
    df_document_cu['finish'] = g1_dict['list_finish'] 
    df_document_cu.to_csv('../result_data/'+'df_document_cut.csv', index = False)
    
    
    
    df_document_in['name'] = g2_dict['list_namec']
    df_document_in['come'] = g2_dict['list_come'] 
    df_document_in['wait'] = g2_dict['list_wait']  
    df_document_in['begin'] = g2_dict['list_begin'] 
    df_document_in['finish'] = g2_dict['list_finish']  
    df_document_in.to_csv('../result_data/'+'df_document_input.csv', index = False)
    
    
    df_document_re['name'] = g3_dict['list_namec']
    df_document_re['come'] = g3_dict['list_come'] 
    df_document_re['wait'] = g3_dict['list_wait']  
    df_document_re['begin'] = g3_dict['list_begin'] 
    df_document_re['finish'] = g3_dict['list_finish']  
    df_document_re.to_csv('../result_data/'+'df_document_review.csv', index = False)           
            
    return df_document_cl,df_document_cu,df_document_in,df_document_re

def get_huanjie_data(classify_data,cut_data,input_data,review_data,capacity_list):
    #2  生成环节数据
    df_huanjie_cl = pd.DataFrame()
    df_huanjie_cu = pd.DataFrame()
    df_huanjie_in = pd.DataFrame()
    df_huanjie_re = pd.DataFrame()
    
    df_huanjie_cl['time'] = [i[0] for i in classify_data]  
    df_huanjie_cl['users'] = [i[1] for i in classify_data]  
    df_huanjie_cl['queue'] = [i[2] for i in classify_data]  
    
    df_huanjie_cu['time'] = [i[0] for i in cut_data]  
    df_huanjie_cu['users'] = [i[1] for i in cut_data]  
    df_huanjie_cu['queue'] = [i[2] for i in cut_data]  
    
    df_huanjie_in['time'] = [i[0] for i in input_data]  
    df_huanjie_in['users'] = [i[1] for i in input_data]  
    df_huanjie_in['queue'] = [i[2] for i in input_data]  
    
    df_huanjie_re['time'] = [i[0] for i in review_data]  
    df_huanjie_re['users'] = [i[1] for i in review_data]  
    df_huanjie_re['queue'] = [i[2] for i in review_data]  
    

                
    df_huanjie_cl = complete_df(df_huanjie_cl,capacity_list[0])
    df_huanjie_cu = complete_df(df_huanjie_cu,capacity_list[1])
    df_huanjie_in = complete_df(df_huanjie_in,capacity_list[2])
    df_huanjie_re = complete_df(df_huanjie_re,capacity_list[3])
    
    
    
    df_huanjie_cl.to_csv('../result_data/df_huanjie_classify.csv', index = False)
    df_huanjie_cu.to_csv('../result_data/df_huanjie_cut.csv', index = False)
    df_huanjie_in.to_csv('../result_data/df_huanjie_input.csv', index = False)
    df_huanjie_re.to_csv('../result_data/df_huanjie_review.csv', index = False)



def result_analyse(action,df_document_cl,df_document_cu,df_document_in,df_document_re):
    df_document_cl.columns = ['name','come_cl','wait_cl','begin_cl','finish_cl','t_relax_time']
    df_document_cu.columns = ['name','come_cu','wait_cu','begin_cu','finish_cu']
    df_document_in.columns = ['name','come_in','wait_in','begin_in','finish_in']
    df_document_re.columns= ['name','come_re','wait_re','begin_re','finish_re']
    
    
    #合并数据
    df_total_process = pd.merge(df_document_cl[['name','come_cl','wait_cl','t_relax_time']],df_document_cu[['name','wait_cu']],how ='left',on = ['name'])
    
    df_total_process = pd.merge(df_total_process,df_document_in[['name','wait_in']],how ='left',on = ['name'])
    df_total_process = pd.merge(df_total_process,df_document_re[['name','wait_re','finish_re']],how ='left',on = ['name'])
    
    #df_total_process = pd.merge(df_document_cl,df_document_cu,how ='left',on = ['name'])
    
    #df_total_process['t_relax_time']=30#测试用，后期删掉
    
    #得到超时率
    df_total_process['out_time'] = df_total_process['come_cl'] + df_total_process['t_relax_time'] -  df_total_process['finish_re'] 
    
    df_total_process['out_time'][df_total_process['out_time']>=0] = 0
    df_total_process['out_time'][df_total_process['out_time']<0] = 1
    out_rate_t = []
    
    for i in range(NUM_BATCH):
        tmp_out_rate_t = np.mean(df_total_process['out_time'][:(i+1)*BATCH_ORDERS])
        out_rate_t.append(tmp_out_rate_t)
    
    return out_rate_t

#    #进行绘制
#    plt.plot(range(len(out_rate_t)), out_rate_t,  color='orange', label=action)
#    plt.legend() # 显示图例
#    plt.xlabel('iteration times')
#    plt.ylabel('out_time')
#    plt.show()



