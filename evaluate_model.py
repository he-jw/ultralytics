from ultralytics import YOLO

# 加载你训练好的最佳模型
model = YOLO('runs/detect/bottle_detection_yolo11n/weights/best.pt')

# 验证模型在验证集上的性能
print("正在验证集上评估模型...")
metrics = model.val(
    data='datasets/bottle/data.yaml',
    split='val'  # 指定使用验证集
)

print("\n--- 验证集评估结果 ---")
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP75: {metrics.box.map75}")


# (推荐) 验证模型在测试集上的性能，这能更好地反映模型的泛化能力
print("\n正在测试集上评估模型...")
metrics_test = model.val(
    data='datasets/bottle/data.yaml',
    split='test'  # 指定使用测试集
)

print("\n--- 测试集评估结果 ---")
print(f"mAP50-95: {metrics_test.box.map}")
print(f"mAP50: {metrics_test.box.map50}")
print(f"mAP75: {metrics_test.box.map75}") 