from pathlib import Path

from .utils import frame_cache_rel_path, load_pickle, save_pickle


def save_frame_cache(cache_root, frame_data):
    cache_root = Path(cache_root)
    rel_path = frame_cache_rel_path(frame_data['sequence_id'], frame_data['frame_idx'])
    save_pickle(cache_root / rel_path, frame_data)
    return rel_path


def load_frame_cache(cache_root, sequence_id, frame_idx):
    cache_root = Path(cache_root)
    return load_pickle(cache_root / frame_cache_rel_path(sequence_id, frame_idx))


def save_cache_index(cache_root, index_data):
    save_pickle(Path(cache_root) / 'cache_index.pkl', index_data)


def load_cache_index(cache_root):
    return load_pickle(Path(cache_root) / 'cache_index.pkl')
