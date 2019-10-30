# -*- coding: utf-8 -*-
import os
import csv
import xlrd
from xlutils.copy import copy;
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#factory
import random
import simpy
import time
from functools import partial, wraps
#sim_index
import re
from decimal import getcontext, Decimal


def  callFromGams(capacity_list):
    
    #运用模型计算出对应预测结果，得到optimize表
    '''
    
    ----------Model hcq----------
    '''
        
    #四个环节的处理时间的模拟
    dforig_input = pd.read_csv('input_data/sim_input.csv', encoding = 'utf-8')
    dforig_classify = pd.read_csv('input_data/sim_classify.csv', encoding = 'utf-8')
    dforig_cut = pd.read_csv('input_data/sim_cut.csv', encoding = 'utf-8')
    dforig_review = pd.read_csv('input_data/sim_review.csv', encoding = 'utf-8')
    
    
    #一变多的分布
    df_caltocut_distr = pd.read_csv('input_data/caltocut_distribution.csv', encoding = 'utf-8')
    
#    capacity_list = [1,6,1,2]
    
    RANDOM_SEED = 42
    NEW_CUSTOMERS = 100 # Total number of customers
    #INTERVAL_CUSTOMERS = 12.0  #到达时间间隔   1/lamda  Generate new customers roughly every x seconds
    MIN_PATIENCE = 999999  # Min. customer patience
    MAX_PATIENCE = 1000000  # Max. customer patience
    time_in_fac=34.0   #服务时间间隔，废弃，暂时不用
    lamda = 1/10#到达时间服从指数分布,此处的t为间隔时间,1/10为lamda,均值为10
    GENERATION_0 = '00classify'
    GENERATION_1 = '01cut'
    GENERATION_2 = '02input'
    GENERATION_3 = '03review'
    g0_list_come = []
    g0_list_begin = []
    g0_list_wait = []
    g0_list_finish = []
    g0_list_name = []
    
    
    g1_list_come = []
    g1_list_begin = []
    g1_list_wait = []
    g1_list_finish = []
    g1_list_name = []
    
    g2_list_come = []
    g2_list_begin = []
    g2_list_wait = []
    g2_list_finish = []
    g2_list_name = []#暂时不用
    
    g3_list_come = []
    g3_list_begin = []
    g3_list_wait = []
    g3_list_finish = []
    g3_list_name = []#暂时不用
    
    g4_list_come = []
    sum_cut_number_list = []
    
    
    #0402添加
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
    
    def source_cal(env,
                   number, 
                   counter,
                   generation,
                   generation_list_begin,
                   generation_list_wait,
                   generation_list_finish,
                   df_simtime,
                   generation_list_name):
    #    global g0_list_come
        sum_num = 0
        """Source generates Doc randomly"""
        for i in range(number):
            serve_time = np.random.choice(df_simtime['sim_time'])#得到模拟数据
    #        print(serve_time)
            c = document(env,
                         'Doc%02d' % i,
                         generation, 
                         counter, 
                         time_in_fac,
                         generation_list_begin,
                         generation_list_wait,
                         generation_list_finish,
                         serve_time,
                         generation_list_name)
            env.process(c)
            t = random.expovariate(lamda)#到达时间服从指数分布,此处的t为间隔时间,1/10为lamda,均值为10
            g0_list_come.append(sum_num)
            sum_num += t 
    
            yield env.timeout(t)
            
#        return g0_list_come
    
    
    #在切的这里进行一变多      
    def source_cut(env, 
                   number, 
                   counter,
                   generation,
                   generation_list_come,
                   generation_list_wait,
                   generation_list_begin,
                   generation_list_finish,
                   df_simtime,
                   generation_list_name,
                   sum_cut_number_list):   
        """Source generates Doc randomly"""
        sum_cut_number = 0
        for i in range(number):
            sample_j  = np.random.choice(df_caltocut_distr['time'])
            sum_cut_number += sample_j
            for j in range(sample_j):
                if j == 0:
                    if i == 0:
                        t = generation_list_come[i]#到达时间服从指数分布,此处的t为间隔时间
                    else:
                        t = generation_list_come[i] - generation_list_come[i-1]
                else:
                    t = 0
                    
                yield env.timeout(t)
                serve_time = np.random.choice(df_simtime['sim_time'])#得到模拟数据
    #            print(serve_time)
                c = document(env, 
                             'Doc%02d_%02d' %(i,j), 
                             generation,
                             counter, 
                             time_in_fac,
                             generation_list_begin,
                             generation_list_wait,
                             generation_list_finish,
                             serve_time,
                             generation_list_name)
                env.process(c)
        sum_cut_number_list.append(sum_cut_number)
    def source_input(env, 
                     number, 
                     counter,
                     generation,
                     generation_list_come,
                   generation_list_wait,
                   generation_list_begin,
                   generation_list_finish,
                     df_simtime,
                     generation_list_name,
                     g1_list_name):   
#        global g1_list_name
        """Source generates Doc randomly"""
        for i in range(number):
            if i == 0:
                t = generation_list_come[i]#到达时间服从指数分布,此处的t为间隔时间
            else:
                t = generation_list_come[i] - generation_list_come[i-1]
            yield env.timeout(t)
            serve_time = np.random.choice(df_simtime['sim_time'])#得到模拟数据
    #        print(serve_time)
            c = document(env, 
                         g1_list_name[i], 
                         generation, 
                         counter, 
                         time_in_fac,
                         generation_list_begin,
                         generation_list_wait,
                         generation_list_finish,
                         serve_time,
                         generation_list_name)
            env.process(c)
    
    def source_review(env, 
                      number, 
                      counter,
                      generation,
                     generation_list_come,
                   generation_list_wait,
                   generation_list_begin,
                   generation_list_finish,
                      df_simtime,
                      generation_list_name,g2_list_name):   
        """Source generates Doc randomly"""
#        global g2_list_name
        for i in range(number):
            if i == 0:
                t = generation_list_come[i]#到达时间服从指数分布,此处的t为间隔时间
            else:
                t = generation_list_come[i] - generation_list_come[i-1]
            yield env.timeout(t)
            serve_time = np.random.choice(df_simtime['sim_time'])#得到模拟数据
    #        print(serve_time)
            c = document(env, g2_list_name[i], generation, counter, time_in_fac,generation_list_begin,generation_list_wait,generation_list_finish,serve_time,generation_list_name)
            env.process(c)
    
    def document(env, 
                 name, 
                 generation,
                 counter,
                 time_in_fac,
                 generation_list_begin,
                 generation_list_wait,
                 generation_list_finish,
                 serve_time,
                 generation_list_name):
        """Doc arrives, is served and leaves."""
        
        arrive = env.now
    #    print('%7.4f %s %s: Here I am' % (arrive,generation, name))
        generation_list_name.append(name)
        
        
        with counter.request() as req:
            patience = random.uniform(MIN_PATIENCE, MAX_PATIENCE)#忍耐时间服从均匀分布
            # Wait for the counter or abort at the end of our tether
            results = yield req | env.timeout(patience)#这句话挺关键的但是我目前没有看懂它是什么意思
            begin_time = env.now
            generation_list_begin.append(begin_time)#添加开始服务时间
            wait = env.now - arrive
            generation_list_wait.append(wait)#添加等待时间
    
            if req in results:
                yield env.timeout(serve_time)
    #            print('%7.4f %s %s: Finished' % (env.now,generation, name))
                finished_time = env.now
                generation_list_finish.append(finished_time)#添加结束时间
                
            else:
                # We reneged
                print('%7.4f %s: RENEGED after %6.3f' % (env.now, name, begin_time))
    
    
    
    # Setup and start the simulation
#    print('Bank renege')
    #random.seed(RANDOM_SEED)
    
    #第0个机器
    env_g0 = simpy.Environment()
    counter = simpy.Resource(env_g0, capacity=capacity_list[0])
    
    classify_data = []
    monitor_cla = partial(monitor, classify_data)
    patch_resource(counter, post=monitor_cla)
    
    env_g0.process(source_cal(env_g0,
                              NEW_CUSTOMERS,
                              counter,
                              GENERATION_0,
                              g0_list_begin,
                              g0_list_wait,
                              g0_list_finish,
                              dforig_classify,
                              g0_list_name))
    env_g0.run()
    g1_list_come = g0_list_finish
    
    
    #第1个机器
    env_g1 = simpy.Environment()
    counter_hcq01 = simpy.Resource(env_g1, capacity=capacity_list[1])
    
    cut_data = []
    monitor_cut = partial(monitor, cut_data)
    patch_resource(counter_hcq01, post=monitor_cut)
    
#    cut_number = source_cut(env_g1, 
#                            NEW_CUSTOMERS, 
#                            counter_hcq01,
#                            GENERATION_1,
#                            g1_list_begin,
#                            g1_list_wait,
#                            g2_list_come,
#                            g1_list_come,
#                            dforig_cut,
#                            g1_list_name)
    env_g1.process(source_cut(env_g1, 
                              NEW_CUSTOMERS, 
                              counter_hcq01,
                              GENERATION_1,
                              g1_list_come,
                              g1_list_wait,
                              g1_list_begin,
                              g1_list_finish,
                              dforig_cut,
                              g1_list_name,
                              sum_cut_number_list))
    env_g1.run()
    g2_list_come = g1_list_finish
    sum_cut_number = sum_cut_number_list[0]#一变多之后的总数量
    
    #第2个机器
    env_g2 = simpy.Environment()
    counter_hcq02 = simpy.Resource(env_g2, capacity=capacity_list[2])
    input_data = []
    monitor_input = partial(monitor, input_data)
    patch_resource(counter_hcq02, post=monitor_input)
    
    env_g2.process(source_input(env_g2, 
                                sum_cut_number,
                                counter_hcq02,
                                GENERATION_2,
                                g2_list_come,
                                g2_list_wait,
                                g2_list_begin,
                                g2_list_finish,
                                dforig_input,
                                g2_list_name,
                                g1_list_name))
    env_g2.run()
    g3_list_come = g2_list_finish
    
    
    
    #第3个机器
    env_g3 = simpy.Environment()
    counter_hcq03 = simpy.Resource(env_g3, capacity=capacity_list[3])
    
    review_data = []
    monitor_review = partial(monitor, review_data)
    patch_resource(counter_hcq03, post=monitor_review)
    
    env_g3.process(source_review(env_g3, 
                                 sum_cut_number,
                                 counter_hcq03,
                                 GENERATION_3,
                                 g3_list_come,
                                 g3_list_wait,
                                 g3_list_begin,
                                 g3_list_finish,
                                 dforig_review,
                                 g3_list_name,
                                 g2_list_name))
    env_g3.run()


    
    
    #生成环节数据
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
                
    df_huanjie_cl = complete_df(df_huanjie_cl,capacity_list[0])
    df_huanjie_cu = complete_df(df_huanjie_cu,capacity_list[1])
    df_huanjie_in = complete_df(df_huanjie_in,capacity_list[2])
    df_huanjie_re = complete_df(df_huanjie_re,capacity_list[3])
    
    
    
    
    
    
    #生成文档数据
    df_document_cl = pd.DataFrame()
    
    df_document_cutttttt = pd.DataFrame()
    
    df_document_cu = pd.DataFrame()
    df_document_in = pd.DataFrame()
    df_document_re = pd.DataFrame()
    
    
    df_document_cl['name'] = g0_list_name
    df_document_cl['come'] = g0_list_come
    df_document_cl['wait'] = g0_list_wait 
    df_document_cl['begin'] = g0_list_begin
    df_document_cl['finish'] = g0_list_finish 
    
    
    df_document_cutttttt['name'] = g1_list_name
    df_document_cutttttt['wait'] = g1_list_wait 
    df_document_cutttttt['begin'] = g1_list_begin
    df_document_cutttttt['finish'] = g1_list_finish 
    df_document_cutttttt['come'] = df_document_cutttttt['begin']-df_document_cutttttt['wait']
    #改个输出顺序
    
    df_document_cu['name'] = df_document_cutttttt['name']
    df_document_cu['come'] = df_document_cutttttt['come']
    df_document_cu['wait'] = df_document_cutttttt['wait']
    df_document_cu['begin'] = df_document_cutttttt['begin']
    df_document_cu['finish'] = df_document_cutttttt['finish']
    
    
    
    df_document_in['name'] = g2_list_name
    df_document_in['come'] = g2_list_come
    df_document_in['wait'] = g2_list_wait 
    df_document_in['begin'] = g2_list_begin
    df_document_in['finish'] = g2_list_finish 
    
    
    df_document_re['name'] = g3_list_name
    df_document_re['come'] = g3_list_come
    df_document_re['wait'] = g3_list_wait 
    df_document_re['begin'] = g3_list_begin
    df_document_re['finish'] = g3_list_finish
    
#    print(len(g3_list_come))
    
    
    
    
    
    
    
    
    
    #1.02得到平均逗留时间ws和平均排队等待时间wq
    def get_ws_wq(df_document_cut):
        stay_time_cut = df_document_cut['finish'] - df_document_cut['come']#得到逗留时间队列
        wait_time_cut = df_document_cut['wait']#得到等待时间队列
        ws_cut = np.mean(stay_time_cut)#平均逗留时间
        wq_cut = np.mean(wait_time_cut)#平均排队等待时间
        ws_cut = Decimal.from_float(ws_cut).quantize(Decimal('0.0000'))
        wq_cut = Decimal.from_float(wq_cut).quantize(Decimal('0.0000'))
        return ws_cut,wq_cut
    
    #1.03得到队伍中的队长ls,ls和ss均经过核实，是正确的
    #其次是计算服务强度，时间间隔与其服务人数相乘，再比上满工时的工作量
    def get_ls_ss(df_huanjie_cut,capacity):
        stay_person_cut = list(df_huanjie_cut['users']+df_huanjie_cut['queue'])#得到系统中的总人数，排队加等待
        change_time_cut = list(df_huanjie_cut['time'])#得到时间
        ser_user_cut = list(df_huanjie_cut['users'])#正在服务的人
        sum_time = 0#间隔时间相加即为总时间
        sum_tp = 0 #间隔时间乘以该时段内系统人数，相加之后即为总的人数*时间段，再除以总时间即为ls
        sum_s_s = 0 #间隔时间乘以此时系统正在工作的服务台，相加之后再除以总的时间段，即为系统的服务强度
        for i in range(len(change_time_cut)):
            if i == 0:
                time_delt = change_time_cut[i]
                
            else:
                time_delt = change_time_cut[i] - change_time_cut[i-1]
            sum_time += time_delt#计算总时长
            sum_tp += float(time_delt)*float(stay_person_cut[i-1])#计算ls
            sum_s_s += float(time_delt)*float(ser_user_cut[i-1])#计算服务强度service_strength
        ls_cut = sum_tp/sum_time#系统中单位时刻的平均逗留人数ls
        ser_stre_cut = (sum_s_s/sum_time)/capacity#系统的服务强度ss
        ls_cut = Decimal.from_float(ls_cut).quantize(Decimal('0.0000'))
        ser_stre_cut = Decimal.from_float(ser_stre_cut).quantize(Decimal('0.0000'))
        return ls_cut,ser_stre_cut
    
    ws_cla,wq_cla = get_ws_wq(df_document_cl)
    ls_cla,ser_stre_cla = get_ls_ss(df_huanjie_cl,capacity_list[0])
    #print('this is sim_ws_cla',ws_cla)
    #print('this is sim_wq_cla',wq_cla)
    #print('this is sim_ls_cla',ls_cla)
#    print('sim_ser_stre_cla',ser_stre_cla)
    
    
    ws_cut,wq_cut = get_ws_wq(df_document_cu)
    ls_cut,ser_stre_cut = get_ls_ss(df_huanjie_cu,capacity_list[1])
    #print('this is sim_ws_cut',ws_cut)
    #print('this is sim_wq_cut',wq_cut)
    #print('this is sim_ls_cut',ls_cut)
#    print('sim_ser_stre_cut',ser_stre_cut)
    
    
    ws_inp,wq_inp = get_ws_wq(df_document_in)
    ls_inp,ser_stre_inp = get_ls_ss(df_huanjie_in,capacity_list[2])
    #print('this is sim_ws_inp',ws_inp)
    #print('this is sim_wq_inp',wq_inp)
    #print('this is sim_ls_inp',ls_inp)
#    print('sim_ser_stre_inp',ser_stre_inp)
    
    
    ws_rev,wq_rev = get_ws_wq(df_document_re)
    ls_rev,ser_stre_rev = get_ls_ss(df_huanjie_re,capacity_list[3])
    #print('this is sim_ws_rev',ws_rev)
    #print('this is sim_wq_rev',wq_rev)
    #print('this is sim_ls_rev',ls_rev)
#    print('sim_ser_stre_rev',ser_stre_rev)
    
    mean_ser_stre = float(np.mean([ser_stre_cla,ser_stre_cut,ser_stre_inp,ser_stre_rev]))
    mean_ser_stre = Decimal.from_float(mean_ser_stre).quantize(Decimal('0.0000'))
    #print('均值mean_ser_stre',mean_ser_stre)
    
    
    
    
    
    #1.05得到每个材料的运行总时间并计算单位时间完成的任务数量
    
    rev_time_list = list(df_huanjie_re['time'])
    max_time = max(rev_time_list)
    per_quantity = float(600*100/max_time)
    per_quantity = Decimal.from_float(per_quantity).quantize(Decimal('0.0000'))
    
    
#    print('10min内处理量',per_quantity)
    function_value = per_quantity
    return(function_value)


if __name__ == '__main__':
    a = [1,4,2,3]
    print(callFromGams(a))
    







