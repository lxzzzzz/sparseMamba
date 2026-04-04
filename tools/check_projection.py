import cv2
import numpy as np
import os
import argparse

# ==========================================
# 核心工具类 (严格遵循 KITTI 标准)
# ==========================================
class Calibration:
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = self.read_calib_file(calib_file)
        else:
            calib = calib_file
        
        self.P2 = calib['P2']  
        self.R0 = calib['R0_rect'] 
        self.V2C = calib['Tr_velo_to_cam'] 

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '': continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()]).reshape(3, 4)
                except ValueError:
                    data[key] = np.array([float(x) for x in value.split()]).reshape(3, 3)
        return data

    def cart2hom(self, pts_3d):
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def rect_to_img(self, pts_3d_rect):
        pts_3d_hom = self.cart2hom(pts_3d_rect)
        pts_2d_hom = np.dot(pts_3d_hom, self.P2.T)
        pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2].reshape(-1, 1)
        return pts_2d

    def lidar_to_rect(self, pts_3d_lidar):
        pts_3d_lidar_hom = self.cart2hom(pts_3d_lidar)
        pts_3d_ref = np.dot(pts_3d_lidar_hom, self.V2C.T)
        pts_3d_rect = np.dot(pts_3d_ref, self.R0.T)
        return pts_3d_rect

    def project_lidar_to_image(self, pts_3d_lidar):
        pts_3d_rect = self.lidar_to_rect(pts_3d_lidar)
        return self.rect_to_img(pts_3d_rect)

class Object3d:
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.t = np.array((float(label[11]), float(label[12]), float(label[13])))
        self.ry = float(label[14])

def compute_box_3d(obj, calib):
    """
    针对带有俯仰角的摄像头专属的 KITTI 画框逻辑：
    先将 KITTI 的 Camera Box 退回 LiDAR 坐标系（恢复平坦），建好 8 个顶点后再投影。
    """
    # 1. 提取中心点并还原为几何中心
    loc_cam = obj.t.copy()
    loc_cam[1] -= obj.h / 2.0  
    
    # 2. 从 Camera 系退回 LiDAR 系
    R_v2c = calib.V2C[:, :3]
    T_v2c = calib.V2C[:, 3]
    R_v2c_inv = np.linalg.inv(R_v2c)
    loc_lidar = R_v2c_inv @ (loc_cam - T_v2c)
    
    # 3. 恢复 LiDAR 系下的偏航角 (这就是 OpenPCDet 的底层逻辑)
    yaw_lidar = -obj.ry - np.pi / 2
    
    # 4. 在 LiDAR 系下构建 8 个顶点 (完全平坦)
    l, w, h = obj.l, obj.w, obj.h
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    
    corners_lidar = np.vstack([x_corners, y_corners, z_corners])
    
    # 绕 LiDAR Z轴旋转
    c = np.cos(yaw_lidar)
    s = np.sin(yaw_lidar)
    R_lidar = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    corners_lidar = np.dot(R_lidar, corners_lidar)
    corners_lidar[0, :] += loc_lidar[0]
    corners_lidar[1, :] += loc_lidar[1]
    corners_lidar[2, :] += loc_lidar[2]
    
    # 5. 重新通过含俯仰角的矩阵投影到 Camera 系
    corners_cam = np.dot(R_v2c, corners_lidar) + T_v2c.reshape(3, 1)
    
    return corners_cam.T

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制 3D 框，并用红色高亮车头部分
    """
    qs = qs.astype(np.int32)
    
    # 绘制基础线框
    for k in range(0, 4):
        # 底面
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        # 顶面
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        # 立柱
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        
    # ==========================================
    # 增强：用红色标示车头 (Front Bumper)
    # 根据我们的顶点定义，index 0 和 1 所在的面是车头(X正向)
    # ==========================================
    front_color = (0, 0, 255) # 红色
    # 连接前脸的对角线（画个叉，超级明显）
    cv2.line(image, (qs[0, 0], qs[0, 1]), (qs[5, 0], qs[5, 1]), front_color, thickness)
    cv2.line(image, (qs[1, 0], qs[1, 1]), (qs[4, 0], qs[4, 1]), front_color, thickness)
    # 加粗前脸的底边和顶边
    cv2.line(image, (qs[0, 0], qs[0, 1]), (qs[1, 0], qs[1, 1]), front_color, thickness + 1)
    cv2.line(image, (qs[4, 0], qs[4, 1]), (qs[5, 0], qs[5, 1]), front_color, thickness + 1)

    return image

def main():
    parser = argparse.ArgumentParser()
    # 默认路径请改成你本地实际的路径
    parser.add_argument('--data_root', type=str, default='/home/lx/Vscode_Items/sparseMamba/data/dair_v2x/training', help='Path to KITTI training folder')
    parser.add_argument('--idx', type=str, default='000018', help='Sample index to visualize') 
    args = parser.parse_args()

    # 路径拼接
    img_path = os.path.join(args.data_root, 'image_2', f'{args.idx}.png')
    if not os.path.exists(img_path): 
        img_path = img_path.replace('.png', '.jpg')
        
    label_path = os.path.join(args.data_root, 'label_2', f'{args.idx}.txt')
    calib_path = os.path.join(args.data_root, 'calib', f'{args.idx}.txt')
    pcd_path = os.path.join(args.data_root, 'velodyne', f'{args.idx}.bin')

    print(f"Checking Sample: {args.idx}")
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error reading image: {img_path}")
        return
        
    calib = Calibration(calib_path)
    
    # ------------------------------------------------
    # 1. 投影 LiDAR 点云 (红色点)
    # ------------------------------------------------
    if os.path.exists(pcd_path):
        print("Projecting Point Cloud...")
        points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
        # 只取前方 Z>0 的点，减少计算量
        points = points[points[:, 0] > 0] 
        # 降采样，画得快一点
        points = points[::5] 

        pts_2d = calib.project_lidar_to_image(points[:, :3])
        
        # 过滤掉图像外的点
        h_img, w_img = image.shape[:2]
        mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w_img) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h_img)
        pts_2d = pts_2d[mask]
        
        for pt in pts_2d:
            cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1)
    else:
        print(f"Warning: PCD not found at {pcd_path}")

    # ------------------------------------------------
    # 2. 投影 3D 框 (绿色主体，红色车头)
    # ------------------------------------------------
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            objects = [Object3d(line) for line in lines]

        for obj in objects:
            if obj.cls_type == 'DontCare': continue
            if obj.t[2] < 0.1: continue # 过滤掉相机背后的物体

            corners_3d_cam = compute_box_3d(obj, calib)
            corners_2d = calib.rect_to_img(corners_3d_cam)
            
            # 防越界报错保护
            if np.any(corners_2d < -20000) or np.any(corners_2d > 20000):
                continue
            
            # 类别颜色分配
            color = (0, 255, 0) # 默认绿色 (Car)
            if obj.cls_type == 'Pedestrian': color = (0, 255, 255) # 黄色
            elif obj.cls_type == 'Cyclist': color = (255, 255, 0) # 青色
            elif obj.cls_type == 'Trafficcone': color = (255, 0, 255) # 紫色
            
            image = draw_projected_box3d(image, corners_2d, color=color, thickness=2)
            
            # 写上类别名称
            cv2.putText(image, obj.cls_type, (int(corners_2d[4,0]), int(corners_2d[4,1])-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        print(f"Warning: Label not found at {label_path}")

    # 保存结果
    output_filename = f'diagnosis_{args.idx}.jpg'
    cv2.imwrite(output_filename, image)
    print(f"Saved diagnosis result to {output_filename}")
    print("【图例】红色点 = 激光雷达投影 | 绿色框 = Label主体 | 红色交叉面 = 车头正前方")

if __name__ == '__main__':
    main()