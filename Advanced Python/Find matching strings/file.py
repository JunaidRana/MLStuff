# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:31:14 2018

@author: Junaid.raza
"""

import pandas as pd

df=pd.read_csv("sample.csv",encoding='latin-1')
print (df.head())

print ("===============================")

tags=df['Related']
addr=df['Core Data']


found=0
notfound=0



for number in range(67304):
    
    if tags[number] in addr:
        print ("Valid")
        found=found+1
        df['searchedTags']=tags[number]
        df['foundAddr']=addr[number]
            
    else:
        print ("Invalid")
        notfound=notfound+1
        
print (found )
print (notfound)
print ("==================")
print (df.head())
    
    
    #break
    
   # print (word.str.contains(tags[num],regex=False))
    #if (df['Core Data'].str.contains(tags[num],regex=False)):
        
    #res=df['Core Data'].str.contains(tags[num],regex=False)
    #break

#print (res.describe())
    
