import argparse
import numpy as np
import open3d
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

# --- 统一的视觉风格配置 ---
# 1: Car (绿), 2: Pedestrian (青), 3: Cyclist (黄)
CLASS_COLORS = {
    1: (0, 1, 0), 
    2: (0, 1, 1),       
    3: (1, 1, 0),       
}
WINDOW_NAME = "Ground Truth Visualization (Car/Ped/Cyc)"

def draw_scenes(points, boxes, labels):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name=WINDOW_NAME, width=1024, height=768)
    
    # 1. 统一的点云渲染风格 (黑色背景，彩虹色点云)
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.zeros(3)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    
    # 高度着色
    if points.shape[0] > 0:
        z = points[:, 2]
        min_z, max_z = np.percentile(z, 1), np.percentile(z, 99)
        colors = np.zeros((points.shape[0], 3))
        norm_z = np.clip((z - min_z) / (max_z - min_z + 1e-6), 0, 1)
        # 这种配色方案能清晰显示地面和物体
        colors[:, 0] = 0.5 * norm_z
        colors[:, 1] = 0.5 * norm_z
        colors[:, 2] = 0.8
        pts.colors = open3d.utility.Vector3dVector(colors)
    vis.add_geometry(pts)

    # 2. 绘制 GT 框
    if boxes is not None:
        for i in range(boxes.shape[0]):
            label_id = int(labels[i])
            # 只绘制我们关心的三类
            if label_id in CLASS_COLORS:
                color = CLASS_COLORS[label_id]
                draw_single_box(vis, boxes[i], color)

    vis.run()
    vis.destroy_window()

def draw_single_box(vis, box, color):
    center = box[0:3]
    lwh = box[3:6]
    axis_angles = np.array([0, 0, box[6]])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    line_set.paint_uniform_color(color)
    vis.add_geometry(line_set)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/dair_v2x_models/fusion_voxelnext.yaml')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    
    # 构建数据集
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=0, logger=logger, training=False
    )
    
    print(f"\n======== {WINDOW_NAME} ========")
    print("�� 绿色: Car | �� 青色: Pedestrian | �� 黄色: Cyclist")
    print("按 'Q' 键切换下一帧...\n")

    for idx, batch_dict in enumerate(test_loader):
        points = batch_dict['points'][:, 1:]
        gt_boxes = batch_dict['gt_boxes'][0]
        
        # GT Box 的最后一列是类别 ID
        boxes_coord = gt_boxes[:, :7]
        boxes_labels = gt_boxes[:, 7]

        draw_scenes(points, boxes_coord, boxes_labels)

if __name__ == '__main__':
    main()
