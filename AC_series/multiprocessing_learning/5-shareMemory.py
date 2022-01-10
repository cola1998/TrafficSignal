# ——*coding:utf-8*--
import multiprocess as mp
# 多cpu中只能用共享内存进行交流 放在每个核中间？？？
if __name__ == '__main__':
    '''
    i : 整数
    d : 小数
    '''
    value = mp.Value("i",1) # 先定义一个形式 在传入数据
    array = mp.Array("i",[1,2,3]) # 只能是一维的