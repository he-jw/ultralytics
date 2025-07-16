from ultralytics import YOLO

# 1. 加载自定义 C3CBAM 结构的模型
model = YOLO('ultralytics/cfg/models/11/yolo11_C3CBAM.yaml')  # 指定 C3CBAM 结构

# 注意：由于模型结构改变，不能直接加载原始 yolo11n.pt 预训练权重
# 原因：C3CBAM 模块与原始 C3 模块结构不同，权重 shape 不匹配
# model.load('yolo11n.pt')  # ❌ 这会导致错误

# 2. 开始训练（从头开始训练）
results = model.train(
    data='datasets/bottle/data.yaml',
    epochs=5,
    imgsz=640,
    device='mps',
    batch=16,
    name='bottle_detection_yolo11n_C3CBAM',  # 实验名加后缀
    patience=20,  # 早停耐心值
    save_period=10,  # 每10轮保存一次
    val=True,  # 启用验证
    plots=True,  # 生成训练图表
)

print("训练完成！")
print("最佳模型保存在: runs/detect/bottle_detection_yolo11n_C3CBAM/weights/best.pt")

# 3. (可选) 使用训练好的模型进行预测
# best_model = YOLO('runs/detect/bottle_detection_yolo11n_C3CBAM/weights/best.pt')
# best_model.predict('path/to/your/test_image.jpg', save=True) 