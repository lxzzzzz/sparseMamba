#!/usr/bin/env python3
"""Generate four framework images from the V2X-xian KITTI-style dataset.

The script creates:
1. point cloud input screenshot
2. camera image input crop
3. first-stage point-cloud detection map
4. second-stage tracking result map
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont


@dataclass
class LabelObject:
    frame: int
    track_id: int
    cls_name: str
    center: np.ndarray
    size: np.ndarray
    yaw: float


CLASS_COLORS = {
    "Car": (1.0, 0.72, 0.05),
    "Van": (0.25, 0.78, 1.0),
    "Truck": (1.0, 0.35, 0.25),
    "Pedestrian": (0.35, 1.0, 0.45),
    "Cyclist": (0.85, 0.45, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/media/lx/LY/Roadside/V2X-xian-trainval-kitti",
        help="Dataset root.",
    )
    parser.add_argument("--seq", default="0037", help="Sequence id.")
    parser.add_argument("--frame", type=int, default=18, help="Frame index.")
    parser.add_argument("--out-dir", default=None, help="Output directory.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--trail-frames", type=int, default=10)
    parser.add_argument("--captions", action="store_true", help="Add corner captions to the images.")
    return parser.parse_args()


def read_labels(label_path: Path) -> list[LabelObject]:
    objects: list[LabelObject] = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 17 or fields[2] == "DontCare":
            continue
        frame = int(fields[0])
        track_id = int(fields[1])
        cls_name = fields[2]
        x, y, z, l, w, h, yaw = map(float, fields[10:17])
        objects.append(
            LabelObject(
                frame=frame,
                track_id=track_id,
                cls_name=cls_name,
                center=np.array([x, y, z], dtype=np.float64),
                size=np.array([l, w, h], dtype=np.float64),
                yaw=yaw,
            )
        )
    return objects


def load_points(bin_path: Path) -> np.ndarray:
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    mask = (
        (points[:, 0] > -5.0)
        & (points[:, 0] < 115.0)
        & (points[:, 1] > -25.0)
        & (points[:, 1] < 25.0)
        & (points[:, 2] > -6.0)
        & (points[:, 2] < 6.0)
    )
    return points[mask]


def colorize_points(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    z = points[:, 2]
    x_norm = (x - x.min()) / (np.ptp(x) + 1e-6)
    z_norm = (z - z.min()) / (np.ptp(z) + 1e-6)
    colors = np.stack(
        [
            0.12 + 0.75 * x_norm,
            0.80 - 0.45 * x_norm + 0.10 * z_norm,
            0.95 - 0.60 * x_norm,
        ],
        axis=1,
    )
    return np.clip(colors, 0.05, 1.0)


def make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3].astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(colorize_points(points))
    return pcd


def rotation_from_z_to_vector(direction: np.ndarray) -> np.ndarray:
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if axis_norm < 1e-8:
        if dot > 0:
            return np.eye(3)
        return o3d.geometry.get_rotation_matrix_from_axis_angle([math.pi, 0.0, 0.0])
    axis = axis / axis_norm
    angle = math.acos(dot)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float, color: tuple[float, float, float]):
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return None
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=10)
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color(color)
    cylinder.rotate(rotation_from_z_to_vector(vec), center=(0.0, 0.0, 0.0))
    cylinder.translate((start + end) * 0.5)
    return cylinder


def box_edge_meshes(obj: LabelObject, color: tuple[float, float, float], radius: float = 0.055):
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, obj.yaw])
    obb = o3d.geometry.OrientedBoundingBox(obj.center, rot, obj.size)
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    corners = np.asarray(line_set.points)
    meshes = []
    for start_id, end_id in np.asarray(line_set.lines):
        mesh = cylinder_between(corners[start_id], corners[end_id], radius, color)
        if mesh is not None:
            meshes.append(mesh)
    return meshes


def track_color(track_id: int) -> tuple[float, float, float]:
    palette = [
        (1.00, 0.38, 0.24),
        (0.25, 0.82, 1.00),
        (0.45, 1.00, 0.38),
        (1.00, 0.86, 0.25),
        (0.78, 0.45, 1.00),
        (1.00, 0.48, 0.78),
        (0.35, 0.62, 1.00),
        (0.70, 1.00, 0.30),
    ]
    return palette[track_id % len(palette)]


def trail_meshes(history: list[np.ndarray], color: tuple[float, float, float], radius: float = 0.10):
    meshes = []
    for p0, p1 in zip(history[:-1], history[1:]):
        start = p0 + np.array([0.0, 0.0, 1.2])
        end = p1 + np.array([0.0, 0.0, 1.2])
        mesh = cylinder_between(start, end, radius, color)
        if mesh is not None:
            meshes.append(mesh)
    if history:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.42, resolution=12)
        sphere.paint_uniform_color(color)
        sphere.translate(history[-1] + np.array([0.0, 0.0, 1.2]))
        meshes.append(sphere)
    return meshes


def text_mesh(label: str, position: np.ndarray, color: tuple[float, float, float]):
    try:
        mesh = o3d.t.geometry.TriangleMesh.create_text(label, depth=0.02).to_legacy()
    except Exception:
        return None
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    scale = 2.2 / max(float(extent[0]), float(extent[1]), 1e-6)
    mesh.scale(scale, center=(0.0, 0.0, 0.0))
    mesh.paint_uniform_color(color)
    mesh.translate(position + np.array([-1.1, 0.7, 2.0]))
    return mesh


def configure_view(vis: o3d.visualization.Visualizer):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.015, 0.018, 0.022])
    opt.point_size = 2.4
    opt.line_width = 4.0
    ctr = vis.get_view_control()
    for _ in range(12):
        ctr.change_field_of_view(step=-5)
    ctr.set_lookat([55.0, 0.0, -0.5])
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.50)


def render_open3d(output_path: Path, width: int, height: int, geometries: list):
    vis = o3d.visualization.Visualizer()
    created = vis.create_window(str(output_path.name), width=width, height=height, visible=True)
    if not created:
        raise RuntimeError("Open3D Visualizer failed to create a window.")
    for geom in geometries:
        vis.add_geometry(geom)
    configure_view(vis)
    for _ in range(30):
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)
    vis.capture_screen_image(str(output_path), do_render=True)
    vis.destroy_window()


def make_image_input(image_path: Path, output_path: Path, width: int, height: int):
    image = Image.open(image_path).convert("RGB")
    src_w, src_h = image.size
    target_ratio = width / height
    crop_w = min(src_w, int(src_h * target_ratio))
    left = (src_w - crop_w) // 2
    image = image.crop((left, 0, left + crop_w, src_h))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(output_path)


def add_corner_caption(image_path: Path, caption: str):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    pad = 18
    box = draw.textbbox((0, 0), caption, font=font)
    text_w = box[2] - box[0]
    text_h = box[3] - box[1]
    draw.rounded_rectangle(
        (pad, pad, pad + text_w + 28, pad + text_h + 22),
        radius=10,
        fill=(5, 8, 14),
        outline=(70, 90, 110),
        width=2,
    )
    draw.text((pad + 14, pad + 10), caption, fill=(235, 240, 245), font=font)
    image.save(image_path)


def overlay_tracking_ids(image_path: Path, objects: list[LabelObject], width: int, height: int):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    # The Open3D view is a top-down BEV. This maps lidar x/y to the rendered canvas
    # closely enough for compact ID tags.
    x_min, x_max = -5.0, 115.0
    y_min, y_max = -25.0, 25.0
    u0, u1 = 250, width - 260
    v0, v1 = height - 210, 190
    for obj in objects:
        x, y, _ = obj.center
        u = int(u0 + (x - x_min) / (x_max - x_min) * (u1 - u0))
        v = int(v0 + (y - y_min) / (y_max - y_min) * (v1 - v0))
        label = f"ID{obj.track_id}"
        box = draw.textbbox((0, 0), label, font=font)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        rgb = tuple(int(c * 255) for c in track_color(obj.track_id))
        draw.rounded_rectangle(
            (u - 6, v - th - 9, u + tw + 8, v + 5),
            radius=5,
            fill=(5, 8, 14),
            outline=rgb,
            width=2,
        )
        draw.text((u + 1, v - th - 5), label, fill=rgb, font=font)
    image.save(image_path)


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "framework_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_name = f"{args.frame:06d}"
    point_path = root / "training" / "velodyne" / args.seq / f"{frame_name}.bin"
    image_path = root / "training" / "image_02" / args.seq / f"{frame_name}.jpg"
    label_path = root / "training" / "label_02" / f"{args.seq}.txt"

    all_objects = read_labels(label_path)
    current_objects = [obj for obj in all_objects if obj.frame == args.frame]
    if not current_objects:
        raise RuntimeError(f"No labels found for seq={args.seq}, frame={args.frame}.")

    histories: dict[int, list[np.ndarray]] = defaultdict(list)
    start_frame = max(0, args.frame - args.trail_frames)
    current_ids = {obj.track_id for obj in current_objects}
    for obj in all_objects:
        if obj.track_id in current_ids and start_frame <= obj.frame <= args.frame:
            histories[obj.track_id].append(obj.center)

    points = load_points(point_path)
    pcd = make_point_cloud(points)

    point_cloud_path = out_dir / "01_point_cloud_input_open3d.png"
    image_input_path = out_dir / "02_image_input.png"
    detection_path = out_dir / "03_detection_output_open3d.png"
    tracking_path = out_dir / "04_tracking_output_open3d.png"

    render_open3d(point_cloud_path, args.width, args.height, [pcd])
    make_image_input(image_path, image_input_path, args.width, args.height)

    detection_geometries = [make_point_cloud(points)]
    for obj in current_objects:
        color = CLASS_COLORS.get(obj.cls_name, (1.0, 0.72, 0.05))
        detection_geometries.extend(box_edge_meshes(obj, color, radius=0.070))
    render_open3d(detection_path, args.width, args.height, detection_geometries)

    tracking_geometries = [make_point_cloud(points)]
    for obj in current_objects:
        color = track_color(obj.track_id)
        tracking_geometries.extend(box_edge_meshes(obj, color, radius=0.075))
        history = sorted(histories[obj.track_id], key=lambda p: float(p[0]))
        tracking_geometries.extend(trail_meshes(history, color, radius=0.11))
        mesh = text_mesh(f"ID{obj.track_id}", obj.center, color)
        if mesh is not None:
            tracking_geometries.append(mesh)
    render_open3d(tracking_path, args.width, args.height, tracking_geometries)
    overlay_tracking_ids(tracking_path, current_objects, args.width, args.height)

    if args.captions:
        add_corner_caption(point_cloud_path, "Point Cloud Input")
        add_corner_caption(image_input_path, "Image Input")
        add_corner_caption(detection_path, "Stage I Detection")
        add_corner_caption(tracking_path, "Stage II Tracking")

    print(f"seq={args.seq}, frame={frame_name}, objects={len(current_objects)}")
    print(f"output_dir={out_dir}")
    for path in [point_cloud_path, image_input_path, detection_path, tracking_path]:
        print(path)


if __name__ == "__main__":
    main()
