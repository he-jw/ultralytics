from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO('yolov10n.pt')

# 2. 开始训练
results = model.train(
    data='datasets/bottle/data.yaml',
    epochs=5,
    imgsz=640,
    device='mps',
    batch=16,
    name='bottle_detection_yolov10n'
)

print("训练完成！")
print("最佳模型保存在: runs/detect/bottle_detection_yolov10n/weights/best.pt")

# 3. (可选) 使用训练好的模型进行预测
# best_model = YOLO('runs/detect/bottle_detection_yolov10n/weights/best.pt')
# best_model.predict('path/to/your/test_image.jpg', save=True) 