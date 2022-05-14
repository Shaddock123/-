cifar100数据集放在data文件夹里（由于是公用数据集且较大，未上传）
使用命令：python train.py 训练模型
使用命令：python test.py    进行模型测试
其他文件：均为被调用的文件，根据函数即可知道具体被调用的哪个模型
训练、测试中的各种曲线使用tensorboard保存在了log文件夹里，训练好的模型保存在了runs/TEST文件夹里（已上传百度网盘，不在这里展示）

网络结构：resnet18， batch size：128，  learning rate：0.1，  优化器：SGD优化器   epoch：80，  loss function： 交叉熵损失函数
    评价指标：准确率
