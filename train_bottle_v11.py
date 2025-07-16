from ultralytics import YOLO

# 1. 加载预训练模型
# 选择一个适合你任务的模型。yolo11n.pt 是最小最快的模型，适合快速入门。
# 其他选项有 yolo11s, yolo11m, yolo11l, yolo11x。
model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')  # 先指定结构
model.load('yolo11n.pt')  # 再加载权重

# 2. 开始训练
# - data: 你的数据集配置文件路径。
# - epochs: 训练轮数，可以先从一个较小的值开始，比如50，然后根据效果增加。
# - imgsz: 训练图片的尺寸。
# - device: 使用 'mps' 来利用 M2 Mac 的 GPU 加速。
# - batch: 每个批次的图片数量，可以根据你的 GPU 内存调整。如果遇到内存不足的错误，可以减小这个值。
# - name: 训练任务的名称，结果会保存在 'runs/detect/{name}' 目录下。
results = model.train(
    data='datasets/Flow/data.yaml',
    epochs=5,
    imgsz=640,
    device='mps',
    batch=16,
    name='Garbage_detection_yolo11n'
)

# 训练完成后，最佳模型会保存在 'runs/detect/bottle_detection_yolo11n/weights/best.pt'
print("训练完成！")
print("最佳模型保存在: runs/detect/Garbage_detection_yolo11n/weights/best.pt")

# 3. (可选) 使用训练好的模型进行预测
# print("使用训练好的模型进行预测...")
# best_model = YOLO('runs/detect/bottle_detection_yolo11n/weights/best.pt')
# best_model.predict('path/to/your/test_image.jpg', save=True) 