# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:56:13 2018

@author: Junaid.raza
"""

from gtts import gTTS
import speech_recognition as sr
import os
import webbrowser
import smtplib

def talkToMe(audio):
    print (audio)
    tts=gTTS(text=audio,lang='en')
    tts.save('audio.mp3')
    os.system('mpg123 audio.mp3')

#Listen for Command
def myCommand():
    r= sr.Recognizer()
    with sr.Microphone() as source:
        print ('I am ready for your next command.')
        r.pause_threshold=1
        r.adjust_for_ambient_noise(source,duration=1)
        audio= r.listen(source)
        
        
    try:
        command=r.recognize_google(audio)
        print ('You said '+command+ '/n')
    #Loop back to continue to listen from commands
    
    except sr.UnknownValueError:
        assistant(myCommand())
        
    return command
         
#Executing commands
def assistant(command):
    if 'open google' in command:
        chrome_path='C:/Program Files (x86)/Google/Chrome/'
        url='https://www.google.com/'
        webbrowser.get(chrome_path).open(url)
        
    if 'hello' in command:
        talkToMe('Hey')
    if 'email' in command:
        talkToMe('who is the recipient')
        recipient=myCommand()
        
        if 'john' in recipient:
            talkToMe('what should i say')
            content=myCommand()
            #init gmail smtp
            mail=smtplib.SMTP('smtp.gmail.com',587)
            #Identity to server
            mail.ehlo()
            #encrypt session
            mail.starttls()
            #mail login
            #Enter your login credentials to init session
            mail.login('user@gmail.com', 'password')
            #Send mail to
            mail.sendmail('Person Name','junaidraza52@gmail.com',content)
            #Close connection
            mail.close()
            
            talkToMe('Email Sent')
            
talkToMe('I am ready for your command:')
#To keep in loop commands
while True:
    assistant(myCommand())
        
    















