from tensorboardX import SummaryWriter
# scalar
# writer = SummaryWriter('runs/scalar_example')  # 写入的目录
# for i in range(10):
#     writer.add_scalar('quadratic',i**2,global_step=i)
#     writer.add_scalar('exponential',2**i,global_step=i)
#
# writer = SummaryWriter('runs/another_scalar_example') # 在另一个路径中写
# for i in range(10):
#     writer.add_scalar('quadratic',i**3,global_step=i)
#     writer.add_scalar('exponential',3**i,global_step=i)

# 直方图 histogram
import numpy as np
writer = SummaryWriter('runs/embedding_example')
writer.add_histogram('normal_centered',np.random.normal(0,1,1000),global_step=1)
writer.add_histogram('normal_centered',np.random.normal(0,2,1000),global_step=50)
writer.add_histogram('normal_centered',np.random.normal(0,3,1000),global_step=100)

# graph
