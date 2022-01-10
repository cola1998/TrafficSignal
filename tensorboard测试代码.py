import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='test_tensorboard')  # 用于记录要可视化的数据

for x in range(100):
    writer.add_scalar('y=2x', x * 2, x)  # 'y=2x'是标量的名称， x*2是曲线的y轴，x是曲线的x轴
    writer.add_scalar('y=pow(2, x)', 2 ** x, x)

    writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                             "xcosx": x * np.cos(x),
                                             "arctanx": np.arctan(x)}, x)
writer.close()