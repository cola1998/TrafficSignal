# --* coding:utf-8 * --
import multiprocess as mp  # 多进程
import threading as td
import time

def job(q):  # 不能有return
    res = 0
    for i in range(100000):
        res += i + i ** 2 + i ** 3
    q.put(res)

def multicore():
    q = mp.Queue()  # multiprocess的queue
    p1 = mp.Process(target=job, args=(q,))
    p2 = mp.Process(target=job, args=(q,))
    p1.start()  # 开始工作
    p2.start()

    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()  # 分两批输出
    # print("multi core: ",res1+res2)


def normal():
    # 普通方式计算
    res = 0
    for _ in range(2):
        for i in range(100000):
            res += i + i ** 2 + i ** 3
    # print("normal: ",res)

def multithread():
    q = mp.Queue()
    t1 = td.Thread(target=job, args=(q,))
    t2 = td.Thread(target=job, args=(q,))
    t1.start()  # 开始工作
    t2.start()

    t1.join()
    t2.join()
    res1 = q.get()
    res2 = q.get()  # 分两批输出
    # print("thread: ",res1+res2)

if __name__ == '__main__':
    start_time = time.time()
    normal()
    st1 = time.time()
    print("normal time = ",st1-start_time)
    multicore()
    st2 = time.time()
    print("multi core time = ", st2 - st1)
    multithread()
    st3 = time.time()
    print("multi thread time = ", st3 - st2)