# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:12:28 2018

@author: Junaid.raza
"""

class Shark:
    
    def __init__(self):
        print("This is the constructor method.")
        
        
    def swim(self):
        print("The shark is swimming.")

    def be_awesome(self):
        print("The shark is being awesome.")


def main():
    sammy = Shark()
    sammy.swim()
    sammy.be_awesome()

if __name__ == "__main__":
    main()