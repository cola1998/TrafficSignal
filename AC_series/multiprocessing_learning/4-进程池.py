# -- * coding:utf-8 * --
import multiprocess as mp

def job(x):
    return x*x

def multicore():
    '''
    map() 可以接受多个数据 自动分配计算机内核 同时计算
    apply_async()  一次只能在一个进程中算一次
    :return:
    '''
    pool = mp.Pool()  # 进程池  默认使用全部核  processes=2 指定使用两个核
    res = pool.map(job,range(10))  # 运算值 自动分配进程等等 map可以自动迭代
    print(res)

    res = pool.apply_async(job,(2,))  # 只能输入一个值
    print(res.get())

    multi_res = [pool.apply_async(job,(i,)) for i in range(10)] # 多个时候 循环获取
    print([res.get() for res in multi_res])

if __name__ == '__main__':
    multicore()