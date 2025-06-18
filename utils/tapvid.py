# ==== Below is based on https://github.com/facebookresearch/co-tracker/blob/main/cotracker/datasets/tap_vid_datasets.py ====
import os
import io
from glob import glob
import torch
import pickle
import numpy as np
import cv2
from PIL import Image
from typing import Mapping, Tuple, Union
import torch.nn.functional as F
import time

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]

# def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
#     """Resize a video to output_size."""
#     # If you have a GPU, consider replacing this with a GPU-enabled resize op,
#     # such as a jitted jax.image.resize.  It will make things faster.
#     # First, resize each frame to 256x256:
#     video_resized_256 = np.stack([cv2.resize(frame, (256, 256)) for frame in video])

#     # Then, resize each frame from 256x256 to 720x480:
#     video_resized = np.stack([cv2.resize(frame, (output_size[1], output_size[0])) for frame in video_resized_256])
#     return video_resized

def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a video (T, H, W, C) to output_size using GPU with torch.

    First resize each frame to (256, 256), then to output_size (H, W) (e.g., 480, 720).
    """
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float()

    # First resize to 256x256
    video_resized_256 = F.interpolate(video_tensor, size=(256, 256), mode='bilinear', align_corners=False)

    # Then resize to output size (H, W)
    video_resized_final = F.interpolate(video_resized_256, size=output_size, mode='bilinear', align_corners=False)
    video_original = F.interpolate(video_tensor, size=(480, 720), mode='bilinear', align_corners=False)

    return video_resized_final, video_original


def resize_video_high_resol(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""

    # Then, resize each frame from 256x256 to 720x480:
    video_resized = np.stack([cv2.resize(frame, (output_size[1], output_size[0])) for frame in video])
    return video_resized


def sample_queries_first(
    target_occluded: torch.Tensor,  # (N, T)
    target_points: torch.Tensor,    # (N, T, 2)
    frames: torch.Tensor            # (T, H, W, C)
) -> Mapping[str, torch.Tensor]:

    valid = torch.sum(~target_occluded, dim=1) > 0
    target_points = target_points[valid]
    target_occluded = target_occluded[valid]

    query_points = []
    for i in range(target_points.shape[0]):
        index = torch.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(torch.tensor([index, x, y], device=frames.device))
    query_points = torch.stack(query_points, dim=0)

    return {
        "video": frames.unsqueeze(0),  # (1, T, H, W, C)
        "query_points": query_points.unsqueeze(0),  # (1, N, 3)
        "target_points": target_points.unsqueeze(0),  # (1, N, T, 2)
        "occluded": target_occluded.unsqueeze(0),  # (1, N, T)
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:

    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


class TAPVid(torch.utils.data.Dataset):
    def __init__(self, args):

        data_root = args.tapvid_root
        self.dataset_type = args.eval_dataset
        self.resize_shape = (args.resize_h, args.resize_w)
        self.queried_first = "first" in self.dataset_type

        if "kinetics" in self.dataset_type:
            all_paths = glob(os.path.join(data_root, "*_of_0010.pkl"))
            points_dataset = []
            for pickle_path in all_paths:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                    points_dataset = points_dataset + data
            self.points_dataset = points_dataset

            if args.end == 0:
                self.points_dataset = self.points_dataset[args.start:]
            else:
                self.points_dataset = self.points_dataset[args.start:args.end]


        elif "davis" in self.dataset_type:
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)
                self.video_names = sorted(list(self.points_dataset.keys()))


        elif "rgb_stacking" in self.dataset_type:
            with open(data_root, "rb") as f:
                self.points_dataset = pickle.load(f)

        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))

    def __getitem__(self, index):
        # :return rgbs: (T, 3, 256, 256)       | in range [0, 255]
        # :return trajs: (T, N, 2)             | in range [0, 256-1]
        # :return visibles: (T, N)             | Boolean
        # :return query_points: (N, 3)         | in format (t, y, x), in range [0, 256-1]1

        if "davis" in self.dataset_type:
            video_name = self.video_names[index]
        else:
            video_name = index
        video = self.points_dataset[video_name]
        frames = video["video"].copy()

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])

        frames_ori = None 
        target_points = torch.tensor(self.points_dataset[video_name]["points"].copy())

        if self.resize_shape is not None:
            frames, frames_ori = resize_video(frames, [self.resize_shape[0], self.resize_shape[1]])
            target_points *= torch.tensor([256, 256])  # 1 should be mapped to 256
        else:
            target_points *= torch.tensor([256, 256])

        target_occ = torch.tensor(self.points_dataset[video_name]["occluded"].copy(), dtype=torch.bool)
        converted = sample_queries_first(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = converted["target_points"][0].permute(1, 0, 2).float()  # T, N, D
        visibles = torch.logical_not(converted["occluded"])[0].permute(1, 0) # T, N
        query_points = converted["query_points"][0] # T, N
        query_points[:,1] *= (self.resize_shape[1] / 256)
        query_points[:,2] *= (self.resize_shape[0] / 256)

        return frames, trajs, visibles, query_points, frames_ori

    def __len__(self):
        return len(self.points_dataset)