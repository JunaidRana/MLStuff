# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:25:46 2018

@author: Junaid.raza
"""

"""
CHAID can be used for Prediction, Classification, and Interection detection.

x*2=sum(observed-expected)*2/expected

x*2 is chai square

CHAID is a tool used to discover the relationship between variables.  
CHAID analysis builds a predictive medel, or tree, to help determine 
how variables best merge to explain the outcome in the given dependent variable. 


In  CHAID analysis, nominal, ordinal, and continuous data can be used, 
where continuous predictors are split into categories with approximately 
equal number of observations.  CHAID creates all possible cross tabulations 
for each categorical predictor until the best outcome is achieved and no 
further splitting can be performed. 


The development of the decision, or classification tree, 
starts with identifying the target variable or dependent variable; 
which would be considered the root.  CHAID analysis splits the target 
into two or more categories that are called the initial, or parent nodes, 
and then the nodes are split using statistical algorithms into child nodes. 
Unlike in regression analysis, the CHAID technique does not require the data 
to be normally distributed.


Merging: 
    In CHAID analysis, if the dependent variable is continuous, the F test is used and if the dependent variable is categorical, the chi-square test is used.  Each pair of predictor categories are assessed to determine what is least significantly different with respect to the dependent variable.  Due to these steps of merging, a Bonferroni adjusted p-value is calculated for the merged cross tabulation.

Decision tree components in CHAID analysis:

In CHAID analysis, the following are the components of the decision tree:

Root node: 
    Root node contains the dependent, or target, variable.  For example, 
    CHAID is appropriate if a bank wants to predict the credit card risk based 
    upon information like age, income, number of credit cards, etc.  
    In this example, credit card risk is the target variable and the remaining 
    factors are the predictor variables.
Parent’s node: 
    The algorithm splits the target variable into two or more categories.  
    These categories are called parent node or initial node.  For the bank example, 
    high, medium and low categories are the parent’s nodes.
Child node: 
    Independent variable categories which come below the parent’s categories 
    in the CHAID analysis tree are called the child node.
Terminal node: 
    The last categories of the CHAID analysis tree are called the terminal node.  
    In the CHAID analysis tree, the category that is a major influence on the 
    dependent variable comes first and the less important category comes last.  
    Thus, it is called the terminal node.
"""

#Creating a CHAID Tree

from CHAID import Tree
import numpy as np
import pandas as pd

print("hello")

## create the data
ndarr = np.array(([1, 2, 3] * 5) + ([2, 2, 3] * 5)).reshape(10, 3)
df = pd.DataFrame(ndarr)
df.columns = ['a', 'b', 'c']
arr = np.array(([1] * 5) + ([2] * 5))
df['d'] = arr

print (df)

## set the CHAID input parameters
independent_variable_columns = ['a', 'b', 'c']
dep_variable = 'd'

## create the Tree via pandas
tree = Tree.from_pandas_df(df, dict(zip(independent_variable_columns, ['nominal'] * 3)), dep_variable)
## create the same tree, but without pandas helper
tree = Tree.from_numpy(ndarr, arr, split_titles=['a', 'b', 'c'], min_child_node_size=5)
## create the same tree using the tree constructor
"""
cols = [
  NominalColumn(ndarr[:,0], name='a')
  NominalColumn(ndarr[:,1], name='b')
  NominalColumn(ndarr[:,2], name='c')
]
tree = Tree(cols, NominalColumn(arr, name='d'), {'min_child_node_size': 5})

"""

print (tree.print_tree())

## to get a LibTree object,
print (tree.to_tree())

## the different nodes of the tree can be accessed like
first_node = tree.tree_store[0]

print (first_node)

## the properties of the node can be access like
print (first_node.members)

## the properties of split can be accessed like
print (first_node.split.p)

print (first_node.split.score)


#Creating a Tree using Bartlett's or Levene's Significance Test for Continuous Variables

"""
When the dependent variable is continuous, the chi-squared test does not work
due to very low frequencies of values across subgroups. As a consequence, 
and because the F-test is very susceptible to deviations from normality, 
the normality of the dependent set is determined and Bartlett's test for 
significance is used when the data is normally distributed (although the subgroups 
may not necessarily be so) or Levene's test is used when the data is non-normal.
"""

## create the data
ndarr = np.array(([1, 2, 3] * 5) + ([2, 2, 3] * 5)).reshape(10, 3)
df = pd.DataFrame(ndarr)
df.columns = ['a', 'b', 'c']
df['d'] = np.random.normal(300, 100, 10)
independent_variable_columns = ['a', 'b', 'c']
dep_variable = 'd'

print (df)
## create the Tree via pandas
tree = Tree.from_pandas_df(df, dict(zip(independent_variable_columns, ['nominal'] * 3)), dep_variable, dep_variable_type='continuous')

## print the tree (though not enough power to split)
print (tree.print_tree())

#Link to this code
# https://github.com/Rambatino/CHAID































