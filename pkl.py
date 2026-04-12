import pickle
import numpy  # 现在版本已兼容

# 你的文件路径
file_path = "/home/lx/Downloads/tracking_infos_val.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

print("读取成功！", data)