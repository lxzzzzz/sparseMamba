import argparse
import numpy as np
import open3d
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# --- 统一的视觉风格配置 (与 GT 完全一致) ---
# 1: Car (绿), 2: Pedestrian (青), 3: Cyclist (黄)
CLASS_COLORS = {
    1: (0, 1, 0), 
    2: (0, 1, 1),       
    3: (1, 1, 0),       
}
WINDOW_NAME = "Prediction Visualization (Car/Ped/Cyc)"

def draw_scenes(points, boxes, labels, scores):
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name=WINDOW_NAME, width=1024, height=768)

    # 1. 统一的点云渲染风格
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.zeros(3)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    
    if points.shape[0] > 0:
        z = points[:, 2]
        min_z, max_z = np.percentile(z, 1), np.percentile(z, 99)
        colors = np.zeros((points.shape[0], 3))
        norm_z = np.clip((z - min_z) / (max_z - min_z + 1e-6), 0, 1)
        colors[:, 0] = 0.5 * norm_z
        colors[:, 1] = 0.5 * norm_z
        colors[:, 2] = 0.8
        pts.colors = open3d.utility.Vector3dVector(colors)
    vis.add_geometry(pts)

    # 2. 绘制 预测框
    if boxes is not None:
        for i in range(boxes.shape[0]):
            # 过滤低置信度
            if scores[i] < 0.1: 
                continue

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
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/dair_v2x_models/voxelnext_kitti.yaml')
    parser.add_argument('--ckpt', type=str, default='/home/lx/Vscode_Items/sparseMamba/voxelnext_latest_model.pth', help='ckpt path')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=0, logger=logger, training=False
    )
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    print(f"\n======== {WINDOW_NAME} ========")
    print("�� 绿色: Pred Car | �� 青色: Pred Pedestrian | �� 黄色: Pred Cyclist")
    print("注意：如果这里有框但GT里没有，就是误检；颜色不对就是分类错误。")
    print("按 'Q' 键切换下一帧...\n")
    
    with torch.no_grad():
        for idx, batch_dict in enumerate(test_loader):
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model.forward(batch_dict)
            
            points = batch_dict['points'][:, 1:].cpu().numpy()
            
            # 获取预测结果
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            draw_scenes(points, pred_boxes, pred_labels, pred_scores)

if __name__ == '__main__':
    main()