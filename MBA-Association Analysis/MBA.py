# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:35:02 2017

@author: sajan kumar
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import OnehotTransactions

# Example1 demo of using apriori library to calculate support for various itmsets
# having some threshold value.

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]


oht=OnehotTransactions()
data=oht.fit_transform(dataset)

df=pd.DataFrame(data=data,columns=oht.columns_)
df.head()

# calculate support for the defferent intemsets having a supprort value grateter
# then a threshlod (0.6)
support_df=apriori(df,min_support=0.6,use_colnames=True)

# calculate length of the itemset
support_df["length"]=support_df["itemsets"].map(lambda x:len(x))

# get all those itemsets that have atleast 2 items in the itemset
support_df[support_df["length"]>=2]


# Example 2 is the use association rules to derive
# lets load a data-set available related to online retail store 
# http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx

Retail_data=pd.read_excel("Online Retail.xlsx")
Retail_data.head(10)

# Data cleaning 
# 1. Description has blank space
# 2. Invoice No. contains Invoice no starts with letter C
# 3. Remove Transactions having no Invoice Number 

# Intent is to combine the data such that we have 1 record for one transaction 
# having unique Invoice Number and columns having quantity of products purchased 
# by user in that particular order 

Retail_data.dropna(axis=0,subset=["InvoiceNo"],inplace=True)
Retail_data["Description"]=Retail_data["Description"].str.strip()
Retail_data["InvoiceNo"]=Retail_data["InvoiceNo"].astype('str')
Retail_data=Retail_data[~Retail_data["InvoiceNo"].str.contains("C")]

# Create buckets of data for different countries one-by-one and see the 
# Association rules that exists among different set of products 

# Generate list of country with number of transaction that happened in that
# Country
Retail_data["Country"].value_counts()

# Bucket 1 "United Kingdom" - having 487622 Max transactions
bucket_uk=Retail_data[Retail_data["Country"]=="United Kingdom"]
bucket1=(bucket_uk.groupby(["InvoiceNo","Description"])["Quantity"].sum().unstack().\
         reset_index().fillna(0).set_index("InvoiceNo"))

# encode all positive values to 1 and anything else to 0 - OneHotEncoding 

def encode_units(x):
  if x <= 0:
    return 0
  if x >= 1:
    return 1

bucket1 = bucket1.applymap(encode_units)
# drop Postage column
bucket1.drop('POSTAGE', inplace=True, axis=1)

itemset_uk=apriori(bucket1, min_support=0.03, use_colnames=True)
itemset_uk.shape
UK_rules=association_rules(itemset_uk,metric="lift",min_threshold=1)
UK_rules[(UK_rules["confidence"]>0.6)&(UK_rules["lift"]>4)].to_csv("./uk_association_rules.csv")

print bucket1["ROSES REGENCY TEACUP AND SAUCER"].sum()
print bucket1["GREEN REGENCY TEACUP AND SAUCER"].sum()

print bucket1["GREEN REGENCY TEACUP AND SAUCER"].sum()
print bucket1["PINK REGENCY TEACUP AND SAUCER"].sum()

# Bucket 2 "France"

bucket2 = (Retail_data[Retail_data["Country"]=="France"].groupby(["InvoiceNo","Description"])["Quantity"]\
         .sum().unstack().reset_index().fillna(0).set_index("InvoiceNo"))
bucket2_sets = bucket2.applymap(encode_units)

# drop Postage column
bucket2_sets.drop('POSTAGE', inplace=True, axis=1)

itemset_france=apriori(bucket2_sets, min_support=0.07, use_colnames=True)
itemset_france.shape

france_rules=association_rules(itemset_france,metric="lift",min_threshold=1)
france_rules.head()

# filter the rules above some threshold value of confidence and lift
france_rules[(france_rules["lift"]>6) & (france_rules["confidence"]>=0.8)]\
  .to_csv("./france_rules.csv")

print bucket2["ALARM CLOCK BAKELIKE GREEN"].sum()
print bucket2["ALARM CLOCK BAKELIKE RED"].sum()

# Bucket 3 "Germany"

basket3 = (Retail_data[Retail_data['Country'] =="Germany"]\
           .groupby(['InvoiceNo', 'Description'])['Quantity']\
           .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
basket_sets3 = basket3.applymap(encode_units)

basket_sets3.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets3 = apriori(basket_sets3, min_support=0.05, use_colnames=True)
rules3 = association_rules(frequent_itemsets3, metric="lift", min_threshold=1)
rules3[ (rules3['lift'] >= 4) &(rules3['confidence'] >= 0.5)].to_csv("./germany_rules.csv")

basket3["RED RETROSPOT CHARLOTTE BAG"].sum()
basket3["WOODLAND CHARLOTTE BAG"].sum()


