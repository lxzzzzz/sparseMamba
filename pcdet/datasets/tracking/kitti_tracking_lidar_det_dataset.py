import copy
import pickle

import numpy as np

from ...utils import common_utils
from ..dataset import DatasetTemplate


class KittiTrackingLidarDetDataset(DatasetTemplate):
    """
    Detector training dataset for KITTI-tracking-style sequences whose GT boxes are
    already stored in lidar coordinates as annos['gt_boxes_lidar'].
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.det_infos = []
        self.include_data(self.mode)
        default_map = {name: name for name in class_names}
        self.map_class_to_kitti = self.dataset_cfg.get('MAP_CLASS_TO_KITTI', default_map)
        self.class_name_remap = self.dataset_cfg.get('CLASS_NAME_REMAP', {})
        self.default_class_name = self.dataset_cfg.get('DEFAULT_CLASS_NAME', None)

    def _remap_names(self, names):
        if len(names) == 0:
            return names
        return np.asarray([
            self.class_name_remap.get(name, self.default_class_name if self.default_class_name is not None else name)
            for name in names
        ])

    def include_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading lidar detector infos')

        infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_file = self.root_path / info_path
            if not info_file.exists():
                continue
            with open(info_file, 'rb') as f:
                infos.extend(pickle.load(f))

        self.det_infos.extend(infos)

        if self.logger is not None:
            self.logger.info('Total frames for lidar detector dataset: %d', len(infos))

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.det_infos) * self.total_epochs
        return len(self.det_infos)

    def get_lidar(self, info):
        lidar_path = self.root_path / info['point_cloud']['lidar_path']
        assert lidar_path.exists(), f'Missing lidar file: {lidar_path}'
        return np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, info['point_cloud'].get('num_features', 4))

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.det_infos)

        info = copy.deepcopy(self.det_infos[index])
        points = self.get_lidar(info)
        input_dict = {
            'frame_id': f"{info.get('sequence_id', 'seq')}_{info.get('frame_id', index)}",
            'points': points,
        }

        if 'annos' in info:
            annos = common_utils.drop_info_with_name(copy.deepcopy(info['annos']), name='DontCare')
            annos['name'] = self._remap_names(annos['name'])
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': np.asarray(annos['gt_boxes_lidar'], dtype=np.float32).reshape(-1, 7),
            })

            get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
            if 'bbox' in annos and 'gt_boxes2d' in get_item_list:
                input_dict['gt_boxes2d'] = annos['bbox']

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if not self.det_infos or 'annos' not in self.det_infos[0]:
            return 'No ground-truth boxes for evaluation', {}

        from ..kitti import kitti_utils
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.det_infos]
        for anno in eval_gt_annos:
            anno['name'] = self._remap_names(anno['name'])

        kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=self.map_class_to_kitti)
        kitti_utils.transform_annotations_to_kitti_format(eval_gt_annos, map_name_to_kitti=self.map_class_to_kitti)

        kitti_class_names = [self.map_class_to_kitti[name] for name in class_names]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict
