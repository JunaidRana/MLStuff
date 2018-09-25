# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:51:27 2018

@author: Junaid.raza
"""

class Dog:
  def say(self):
    print ("hau")

class Cat:
  def say(self):
    print ("meow")

pet = Dog()
pet.say() # prints "hau"
another_pet = Cat()
another_pet.say() # prints "meow"

my_pets = [pet, another_pet]
for a_pet in my_pets:
  a_pet.say()