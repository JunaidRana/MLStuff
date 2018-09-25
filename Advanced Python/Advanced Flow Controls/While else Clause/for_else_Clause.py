# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:07:04 2018

@author: Junaid.raza
"""

numbers=[21,28,43,65,76]
divisor=9

for number in numbers:
    if number % divisor==0:
        found=divisor
        break
    else:
        print ("Not found any number to be divided")
else:
    numbers.append(divisor)
    
print (numbers)