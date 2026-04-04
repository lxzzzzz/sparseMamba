import os
import glob
from collections import Counter
from tqdm import tqdm
import argparse

def count_kitti_classes(label_dir):
    """
    统计 KITTI 格式标签文件中的类别数量
    """
    # 1. 检查路径
    if not os.path.exists(label_dir):
        print(f"错误: 找不到目录 {label_dir}")
        return

    # 2. 获取所有 txt 文件
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    file_count = len(label_files)
    
    if file_count == 0:
        print(f"错误: 在 {label_dir} 下没有找到 .txt 文件")
        return

    print(f"正在分析 {file_count} 个标签文件...")

    class_counter = Counter()
    total_objects = 0

    # 3. 遍历读取
    for file_path in tqdm(label_files, desc="统计中"):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # KITTI 格式: Type Truncated Occluded Alpha x1 y1 x2 y2 h w l x y z ry
                # 我们只需要第一个元素 (Type)
                parts = line.split(' ')
                obj_type = parts[0]
                
                # 过滤掉 DontCare (如果你不想统计它)
                # if obj_type == 'DontCare': continue
                
                class_counter[obj_type] += 1
                total_objects += 1

    # 4. 打印结果
    print("\n" + "="*50)
    print(f"【DAIR-V2X 类别统计报告】")
    print(f"总文件数: {file_count}")
    print(f"总目标数: {total_objects}")
    print(f"包含类别数: {len(class_counter)}")
    print("="*50)
    print(f"{'类别名称 (Class)':<20} | {'数量 (Count)':<10} | {'占比 (Pct)':<10}")
    print("-" * 50)

    # 按照数量从多到少排序
    for cls_name, count in class_counter.most_common():
        percentage = (count / total_objects) * 100
        print(f"{cls_name:<20} | {count:<10} | {percentage:.2f}%")
    print("="*50)

    # 5. 给出配置建议
    print("\n【OpenPCDet 配置文件建议】")
    print("根据上述统计，建议修改配置文件中的 CLASS_NAMES 列表。")
    print("例如:")
    
    # 简单的逻辑：只推荐数量超过一定阈值的类别，或者是标准的KITTI三大类
    kitti_standards = ['Car', 'Pedestrian', 'Cyclist']
    suggested_classes = [name for name, _ in class_counter.most_common() if name in kitti_standards or 'Truck' in name or 'Bus' in name]
    print(f"CLASS_NAMES: {suggested_classes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径：请修改为你转换后的 kitti/training/label_2 路径
    parser.add_argument('--label_dir', type=str, 
                        default='/home/lx/Vscode_Items/sparseMamba/data/dair_v2x/training/label_2',
                        help='Path to label_2 folder')
    
    args = parser.parse_args()
    
    count_kitti_classes(args.label_dir)