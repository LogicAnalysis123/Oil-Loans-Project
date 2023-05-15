# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:57:11 2023

@author: Lenovo
"""

'''分析背景

- 某网约车公司给司机贷款-油品贷的客户badrate明显高，但是现有的风控策略没有显示欺诈，需要进一步分析。
- Leader希望不要上太复杂的模型，希望模型简单，且上的策略效果好。
- 现在每个油品贷的客户都有一个A卡评级-class_new。
- 只有评级是A的客户贷款才是赚钱的，其他的用户坏账率高达5%，是赔钱的（一般坏账率大于3%就是赔钱的）。
- 有许多加油站是和滴滴合作的，加油站给滴滴提供司机的信息，供滴滴做信用评估。所以可以通过加油站提供的信息看看可以在加油上做点什么。
- 目的：在油品贷的细分上想一些办法，让坏账率不那么高
'''
import pandas as pd
import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'c:/users/lenovo/anaconda3/...'
#graphviz的package存放的地点

path = 'D:\\风控\\...'
data = pd.read_excel(path + 'oil_data_for_tree.xlsx')
data.head()


class_list = set(data.class_new)#A卡评级
set(data.bad_ind)


#数据重组:copy()是为了后面改变变量的时候不改变原来的值，这样做错了也不用重新读取数据
org_lst = ['uid','create_dt','oil_actv_dt','class_new','bad_ind']

agg_lst = ['oil_amount','discount_amount','sale_amount','amount',
           'pay_amount','coupon_amount','payment_coupon_amount']

dstc_lst = ['channel_code','oil_code','scene','source_app','call_source']

df = data[org_lst].copy()
df[agg_lst] = data[agg_lst].copy()
df[dstc_lst] = data[dstc_lst].copy()


'''看一下缺失情况
- numpy里边查找NaN值的话，就用np.isnan()
- pandas里边查找NaN值的话，要么.isna()，要么.isnull()
'''

df.isna().sum()
df.describe()

'''不能对原始变量进行随意累加
  
构造变量的时候不能直接对历史所有数据做累加。  
否则随着时间推移，变量分布会有很大的变化。
那么现在做的模型对以后来说就不适用了
'''

#查一下有没有重复的uid
len(set(df.uid))
len(df.uid)

'''对creat_dt做补全，用oil_actv_dt来填补，并且截取6个月的数据。
用激活日期补全初始日期'''
def time_isna(x,y):
    if str(x) == 'NaT':
        x = y
    else:
        x = x
    return x
df2 = df.sort_values(['uid','create_dt'],ascending = False)
df2['create_dt'] = df2.apply(lambda x: time_isna(x.create_dt,x.oil_actv_dt),axis = 1)
df2['dtn'] = (df2.oil_actv_dt - df2.create_dt).apply(lambda x :x.days)
df = df2[df2['dtn']<180]
df.head()

#对org_list变量求历史贷款天数的最大间隔，并且去重
base = df[org_lst]
base['dtn'] = df['dtn']
base = base.sort_values(['uid','create_dt'],ascending = False)#取最新的一条记录
base = base.drop_duplicates(['uid'],keep = 'first')
base.shape

#做变量衍生
gn = pd.DataFrame()
for i in agg_lst:
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:len(df[i])).reset_index()) #len(df[i])有多少行
    tp.columns = ['uid',i + '_cnt'] #+ '_cnt'知道新变量是从那个变量演化来的
    if gn.empty == True:
        gn = tp
    else:
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:np.where(df[i]>0,1,0).sum()).reset_index())
    tp.columns = ['uid',i + '_num']
    if gn.empty == True:
        gn = tp
    else:
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:np.nansum(df[i])).reset_index())
    tp.columns = ['uid',i + '_tot']
    if gn.empty == True:
        gn = tp
    else:
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:np.nanmean(df[i])).reset_index())
    tp.columns = ['uid',i + '_avg']
    if gn.empty == True:
        gn = tp
    else:
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:np.nanmax(df[i])).reset_index())
    tp.columns = ['uid',i + '_max']
    if gn.empty == True:
        gn = tp
    else:
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:np.nanmin(df[i])).reset_index())
    tp.columns = ['uid',i + '_min']
    if gn.empty == True:
        gn = tp
    else:
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:np.nanvar(df[i])).reset_index())
    tp.columns = ['uid',i + '_var']
    if gn.empty == True:
        gn = tp
    else:
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:np.nanmax(df[i]) -np.nanmin(df[i]) ).reset_index())
    tp.columns = ['uid',i + '_var']
    if gn.empty == True:
        gn = tp
    else:
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df:np.nanmean(df[i])/max(np.nanvar(df[i]),1)).reset_index())
    tp.columns = ['uid',i + '_var']
    if gn.empty == True:
        gn = tp
    else:
        gn = pd.merge(gn,tp,on = 'uid',how = 'left')
        
#对dstc_lst变量求distinct个数
gc = pd.DataFrame()
for i in dstc_lst:
    tp = pd.DataFrame(df.groupby('uid').apply(lambda df: len(set(df[i]))).reset_index())
    tp.columns = ['uid',i + '_dstc']
    if gc.empty == True:
        gc = tp
    else:
        gc = pd.merge(gc,tp,on = 'uid',how = 'left')
        
#将变量组合在一起
fn = pd.merge(base,gn,on= 'uid')
fn = pd.merge(fn,gc,on= 'uid') 
fn.shape
len(set(fn.uid))
fn = pd.merge(base,gn,on= 'uid')
fn = pd.merge(fn,gc,on= 'uid') 
fn.shape
fn = fn.fillna(0)#模型里面不能有缺失值，否则无法运行
fn.head(100)

#训练决策树模型
x = fn.drop(['uid','oil_actv_dt','create_dt','bad_ind','class_new'],axis = 1)#这些变量是不能放在模型里做训练的
y = fn.bad_ind.copy()
from sklearn import tree
dtree = tree.DecisionTreeRegressor(max_depth = 2,min_samples_leaf = 500,min_samples_split = 5000)#min_samples_leaf子节点小于500不再做分化
dtree = dtree.fit(x,y)


#输出决策树图像，并作出决策

import pydotplus 
from IPython.display import Image
from six import StringIO
from sklearn import tree
import graphviz
dot_data = StringIO()
tree.export_graphviz(dtree, out_file=dot_data,
                         feature_names=x.columns,
                         class_names=['bad_ind'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())

dot_data = StringIO()
tree.export_graphviz(dtree, out_file=dot_data,
                         feature_names=x.columns,
                         class_names=['bad_ind'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())

sum(fn.bad_ind)/len(fn.bad_ind)

'''坏账率接近5%，是赔钱的；3%以下是赚钱的。
用一个叶节点的规则可以把坏账率控制在2.1%，已经赚钱了
用决策树的两和节点对应的两条规则可以将坏账率控制在1.2%'''

#根据决策树分析结果生成策略
dff1 = fn.loc[(fn.amount_tot > 48077.5)&(fn.discount_amount_cnt > 3.5)].copy()
dff1['level'] = 'oil_A'
dff2 = fn.loc[(fn.amount_tot > 48077.5)&(fn.discount_amount_cnt <= 3.5)].copy()
dff2['level'] = 'oil_B'
dff3 = fn.loc[(fn.amount_tot <= 48077.5)].copy()
dff3['level'] = 'oil_C'

dff1.head()

last = dff1[['class_new', 'level', 'bad_ind', 'uid', 'oil_actv_dt', 'bad_ind']].copy()
last['oil_actv_dt'] = last['oil_actv_dt'].apply(lambda x : str(x)[:7]).copy()
last.sample(5)

#把结果存到excel里面
last.to_excel(path + 'final_report1.xlsx', index = False)