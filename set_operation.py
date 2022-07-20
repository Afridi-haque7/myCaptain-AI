#!/usr/bin/env python
# coding: utf-8

# In[3]:


# MyCaptain Task - 4:
# Write a Python Program to Illustrate Different Set Operations.
# Here we have to define two set variables and we have to perform different set operations: 
# union, intersection, difference and symmetric difference.
# E = {2, 4, 8, 0, 6} N = {1, 2, 3, 4, 5}

E = {0, 2, 4, 6, 8}
N = {1, 2, 3, 4, 5}

# performing union
s1 = E.union(N)
print(s1)
# ans = {0, 1, 2, 3, 4, 5, 6, 8}

#performing intersection
s2 = E.intersection(N)
print(s2)
# ans = {2, 4}

#performing set difference
s3 = E.difference(N)
print(s3)
# ans = {0, 8, 6}

#performing symmetric_difference
s4 = E.symmetric_difference(N)
print(s4)
# ans = {0, 1, 3, 5, 6, 8}


# In[ ]:




