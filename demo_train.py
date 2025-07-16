from torch import device
from ultralytics import YOLO

# 加载预训练的YOLO11n模型
model = YOLO('yolo11n.pt')

# 训练模型
results = model.train(
    data='coco8.yaml',    # 使用内置的示例数据集
    epochs=10,            # 训练轮数
    imgsz=640,           # 图片大小
    verbose=True,        # 显示详细训练信息
    device='mps'         # 使用M1/M2 GPU加速，mps表示Metal Performance Shaders
    # device='cpu'
)

# 验证模型
metrics = model.val()

# 在测试图片上进行预测
results = model('https://ultralytics.com/images/bus.jpg')  # 使用示例图片进行预测