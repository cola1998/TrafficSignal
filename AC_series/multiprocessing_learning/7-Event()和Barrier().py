# ——* coding:utf-8*--
import multiprocess as mp
import time,random

'''
event 描述的是一种同步的处理事件
多个进程拥有一个event实例，调用wait方法将进入到阻塞状态，阻塞标记为False
另外一个进程可以继续工作，并且通过set()方法将阻塞标记置为True
'''

def restaurant_handle(event): # 餐厅的处理进程
    pass

def diners_hangle(event): # 食客的处理进程
    pass

def main():
    event = mp.Event() # 定义一个event同步处理
    resaurant_process = mp.Process(target=restaurant_handle,args=(event,),name="餐厅服务进程")
    diners_process = mp.Process(target=diners_hangle,args=(event,),name='食客进程')