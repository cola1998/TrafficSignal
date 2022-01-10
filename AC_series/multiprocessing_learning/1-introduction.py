# --* coding:utf-8 * --
'''
multiprocessing.Event() | Barrier()
'''

import multiprocess as mp  # 多进程
import threading as td  # 线程


def job(q):  # 不能有return
    res = 0
    for i in range(1000):
        res += i + i ** 2 + i ** 3
    q.put(res)

if __name__ == '__main__':
    # t1 = td.Thread(target=job, args=(1, 2))
    # t1.start()
    # t1.join()  就是谁调用这个方法，就让调用此方法的线程进入阻塞状态，等待我执行完毕之后，再往下执行
    q = mp.Queue()  # multiprocess的queue
    p1 = mp.Process(target=job, args=(q,))
    p2 = mp.Process(target=job, args=(q,))
    p1.start()  # 开始工作
    p2.start()

    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()  # 分两批输出
    print(res1,res2)