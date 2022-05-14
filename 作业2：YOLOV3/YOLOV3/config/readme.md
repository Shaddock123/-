1、将VOC数据集解压到YOLOV3项目的根路径下
2、训练：  使用命令： python train.py 训练模型， 训练好的模型保存在了weight文件夹里,loss曲线以及评价指标保存在了log文件夹里
3、测试：  将要测试的图片放在img文件夹里，使用命令：python test.py --weight_path ./weight/best.pt --visiual ./img 
            检测完的图片保存在results文件里
4、其他文件：均为被调用的文件，根据函数即可知道具体被调用的哪个模型

各种参数的配置在config/yolov3_config_voc.py文件里

网络结构：darknet53， batch size：8，  learning rate： 0.0001， 优化器：SGD，   epoch： 80
loss function：YOLOv3损失函数（包括矩形框中心点误差，预测框宽高误差，预测框置信度损失，预测框类别损失）
评价指标：mAP,mIoU