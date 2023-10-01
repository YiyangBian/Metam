from distutils.ccompiler import new_compiler
import sys
sys.path.append("/Users/yiyangbian/Documents/GitHub/Metam")
import src.backend.profile_weights as profile_weights
import os,copy
from sklearn.feature_selection import mutual_info_classif
from os import listdir
from os.path import isfile, join
import pandas as pd
from src.backend.dataset import Dataset
import math
import pandas as pd
# from src.backend.join_path import JoinKey, JoinPath
from src.backend.join_column import JoinColumn
import pickle
import src.backend.join_path as join_path
import operator,random
from sklearn import datasets, linear_model
import src.backend.group_helper as group_helper
import src.backend.querying as querying

#Oracle implementation, any file containing Oracle class can be used as a task
from src.backend.classifier_oracle import Oracle


random.seed(0)

path='/Users/yiyangbian/Documents/GitHub/Metam/dataset/datasets/basetables'#Add the path to all datasets
query_data='base_school.csv'#Add name of initial dataset
class_attr='Total Student Response Rate (%)'#column name of prediction attribute
query_path=path+"/"+query_data


epsilon=0.05 #Metam parameter
theta = 0.90 #Required utility

uninfo=0 #Number of uninformative profiles to be added on top of default set of profiles

filepath='/Users/yiyangbian/Documents/GitHub/Metam/dataset/sampled_schools.txt'#File containing all join paths


options = join_path.get_join_paths_from_file(query_data,filepath)
# print("options",options)

files = [f for f in listdir(path) if isfile(join(path, f))]


dataset_lst=[]
data_dic={}

base_df=pd.read_csv(query_path)

joinable_lst=options
oracle=Oracle("random forest")
orig_metric=oracle.train_classifier(base_df,'class')

print ("original metric is ",orig_metric)

i=0
new_col_lst=[]
skip_count=0
# print("joinable_lst",joinable_lst)

while i<len(joinable_lst):
    # print ("i,len(joinable_lst)",i,len(joinable_lst))
    jp=joinable_lst[i]
    print (jp.join_path[0].tbl,jp.join_path[0].col,jp.join_path[1].tbl,jp.join_path[1].col)
    
    if jp.join_path[0].tbl not in data_dic.keys():
        # print("1")
        df_l=pd.read_csv(path+"/"+jp.join_path[0].tbl,sep=',',low_memory=False)
        data_dic[jp.join_path[0].tbl]=df_l
        #print ("dataset size is ",df_l.shape)
    else:
        df_l=data_dic[jp.join_path[0].tbl]
    if jp.join_path[1].tbl not in data_dic.keys():
        # print("3")
        # df_r=pd.read_csv(path+"/"+jp.join_path[1].tbl,low_memory=False)
        df_r=pd.read_csv("/Users/yiyangbian/Documents/GitHub/Metam/dataset/datasets/Augmentationsearch/open_data_usa"+"/"+jp.join_path[1].tbl,sep=',',low_memory=False)
        data_dic[jp.join_path[1].tbl]=df_r
        #print ("dataset size is ",df_r.shape)
    else:
        # print("4")
        df_r=data_dic[jp.join_path[1].tbl]

    # print("2131231",df_l,df_r)
    collst=list(df_r.columns)
    # print("nichoucouc",jp.join_path[1].col )
    # print("dsads",jp.join_path[0].col)
    if jp.join_path[1].col not in df_r.columns or jp.join_path[0].col not in df_l.columns:
        # print(("JINLAILE1"))
        i+=1
        continue
        
    for col in collst:

        # print("1",jp,"2",df_r,"3",col,"4",base_df,"5",class_attr,"6",len(new_col_lst),"7",uninfo)
        jc=JoinColumn(jp,df_r,col,base_df,class_attr,len(new_col_lst),uninfo)
        new_col_lst.append(jc)
    i+=1

# print("new_col_lst",new_col_lst)
(centers,assignment,clusters)=join_path.cluster_join_paths(new_col_lst,100,epsilon)
print ("centers",centers)

tau = len(centers)


weights={}
# print("new_col_lst",new_col_lst)
weights=profile_weights.initialize_weights(new_col_lst[0],weights)

metric=orig_metric
initial_df=copy.deepcopy(base_df)
candidates=centers


if tau==1:
    candidates=[i for i in range(len(new_col_lst))]


augmented_df= querying.run_metam(centers,tau,oracle,candidates,theta,metric,initial_df,new_col_lst,weights,class_attr,clusters,assignment,uninfo,epsilon)    
augmented_df.to_csv('augmented_data.csv')

