import visdom
# 新建一个连接客户端
import numpy as np
# 指定env = 'test1'，默认是'main',注意在浏览器界面做环境的切换
vis = visdom.Visdom(env='test1')


#绘制loss变化趋势，参数一为Y轴的值，参数二为X轴的值，参数三为窗体名称，参数四为表格名称，参数五为更新选项，从第二个点开始可以更新
vis.line(Y=np.array([totalloss.item()]), X=np.array([traintime]),
                win=('train_loss'),
                opts=dict(title='train_loss'),
                update=None if traintime == 0 else 'append'
                )
