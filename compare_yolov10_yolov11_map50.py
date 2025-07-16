import pandas as pd
import matplotlib.pyplot as plt
import os

# 新建文件夹保存对比图
output_dir = 'compare_results'
os.makedirs(output_dir, exist_ok=True)

# 读取YOLOv10的训练结果
v10_csv = 'runs/detect/Garbage_detection_yolov10n/results.csv'
df_v10 = pd.read_csv(v10_csv)
map50_v10 = df_v10['metrics/mAP_0.5']

# 读取YOLOv11的训练结果
v11_csv = 'runs/detect/Garbage_detection_yolo11n/results.csv'
df_v11 = pd.read_csv(v11_csv)
map50_v11 = df_v11['metrics/mAP_0.5']

# 取最短长度，防止epoch数不同
min_len = min(len(map50_v10), len(map50_v11))
epochs = range(1, min_len + 1)

plt.plot(epochs, map50_v10[:min_len], 'b', label='YOLOv10')
plt.plot(epochs, map50_v11[:min_len], 'r', label='YOLOv11')
plt.xlabel('epoch')
plt.ylabel('mAP50')
plt.ylim(0, 1.0)
plt.legend()
plt.title('YOLOv10 vs YOLOv11 mAP50')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'yolov10_vs_yolov11_map50.png'), dpi=150)
plt.show()