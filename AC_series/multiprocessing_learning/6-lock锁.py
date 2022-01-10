# --* coding:utf-8 *--
import multiprocess as mp
import time

def job(v,num,l):
    l.acquire()  # 锁住
    for _ in range(10):
        time.sleep(0.1)
        v.value += num
        print("v ",v.value)
    l.release()  # 释放锁

def multicore():
    l = mp.Lock()
    v = mp.Value('i',0)
    p1 = mp.Process(target=job,args=(v,1,l))
    p2 = mp.Process(target=job, args=(v,3,l)) # 观察不加锁时如何抢夺共享内存
    p1.start()
    p2.start() # 基于p1的计算结果继续计算

    p1.join()
    p2.join()

if __name__ == "__main__":
    multicore()