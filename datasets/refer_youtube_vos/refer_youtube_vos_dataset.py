import json
import random
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import torchvision.transforms.functional as F
from os import path
from tqdm import tqdm
from PIL import Image
import numpy as np
from einops import rearrange
import datasets.transforms as T
from misc import nested_tensor_from_videos_list

ytvos_category_dict = {
    'airplane': 0, 'ape': 1, 'bear': 2, 'bike': 3, 'bird': 4, 'boat': 5, 'bucket': 6, 'bus': 7, 'camel': 8, 'cat': 9, 
    'cow': 10, 'crocodile': 11, 'deer': 12, 'dog': 13, 'dolphin': 14, 'duck': 15, 'eagle': 16, 'earless_seal': 17, 
    'elephant': 18, 'fish': 19, 'fox': 20, 'frisbee': 21, 'frog': 22, 'giant_panda': 23, 'giraffe': 24, 'hand': 25, 
    'hat': 26, 'hedgehog': 27, 'horse': 28, 'knife': 29, 'leopard': 30, 'lion': 31, 'lizard': 32, 'monkey': 33, 
    'motorbike': 34, 'mouse': 35, 'others': 36, 'owl': 37, 'paddle': 38, 'parachute': 39, 'parrot': 40, 'penguin': 41, 
    'person': 42, 'plant': 43, 'rabbit': 44, 'raccoon': 45, 'sedan': 46, 'shark': 47, 'sheep': 48, 'sign': 49, 
    'skateboard': 50, 'snail': 51, 'snake': 52, 'snowboard': 53, 'squirrel': 54, 'surfboard': 55, 'tennis_racket': 56, 
    'tiger': 57, 'toilet': 58, 'train': 59, 'truck': 60, 'turtle': 61, 'umbrella': 62, 'whale': 63, 'zebra': 64
}

class ReferYouTubeVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the full
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.
    """
    def __init__(self, subset_type: str = 'train', dataset_path: str = '/data2/hj/RVOS/MTTR/refer_youtube_vos', window_size=12,
                 frame_interval=10,
                 distributed=False, device=None, **kwargs):
        super(ReferYouTubeVOSDataset, self).__init__()
        assert subset_type in ['train', 'test'], "error, unsupported dataset subset type. use 'train' or 'test'."
        if subset_type == 'test':
            subset_type = 'valid'  # Refer-Youtube-VOS is tested on its 'validation' subset (see description above)
        self.subset_type = subset_type
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.videos_dir = path.join(dataset_path, subset_type, 'JPEGImages')
        assert path.isdir(self.videos_dir), f'error: missing frames directory: {self.videos_dir}'
        if subset_type == 'train':
            self.mask_annotations_dir = path.join(dataset_path, subset_type, 'Annotations')  # only available for train
            assert path.isdir(self.mask_annotations_dir), f'error: missing annotations directory: {self.mask_annotations_dir}'
            with open(path.join(dataset_path, subset_type, 'meta.json'), 'r') as f:
                self.subset_metas_by_video = json.load(f)['videos']
        else:
            self.mask_annotations_dir = None
            self.subset_metas_by_video = {}
        self.samples_list = self.generate_samples_metadata(dataset_path, subset_type, window_size, distributed)
        self.transforms = A2dSentencesTransforms(subset_type, **kwargs)
        self.collator = Collator(subset_type)

    def generate_samples_metadata(self, dataset_path, subset_type, window_size, distributed):
        if subset_type == 'train':
            metadata_file_path = './datasets/refer_youtube_vos/train_expression_samples_metadata.json'
        else:  # validation
            metadata_file_path = f'./datasets/refer_youtube_vos/valid_samples_metadata.json'
        if path.exists(metadata_file_path):
            print(f'loading {subset_type} subset samples metadata...')
            with open(metadata_file_path, 'r') as f:
                samples_list = [tuple(a) for a in tqdm(json.load(f), disable=distributed and dist.get_rank() != 0)]
                return samples_list
        elif (distributed and dist.get_rank() == 0) or not distributed:
            print(f'creating {subset_type} subset samples metadata...')
            subset_expressions_file_path = path.join(dataset_path, 'meta_expressions', subset_type, 'meta_expressions.json')
            with open(subset_expressions_file_path, 'r') as f:
                subset_expressions_by_video = json.load(f)['videos']

            if subset_type == 'train':
                samples_list = []
                for vid_id, data in tqdm(subset_expressions_by_video.items()):
                    vid_frames_indices = sorted(data['frames'])
                    for exp_id, exp_dict in data['expressions'].items():
                        if vid_id not in self.subset_metas_by_video:
                            continue
                        if exp_dict['obj_id'] not in self.subset_metas_by_video[vid_id]['objects']:
                            continue
                        exp_record = dict(exp_dict)
                        exp_record['exp_id'] = exp_id
                        samples_list.append((vid_id, vid_frames_indices, exp_record))
            else:  # validation
                # for some reasons the competition's validation expressions dict contains both the validation & test
                # videos. so we simply load the test expressions dict and use it to filter out the test videos from
                # the validation expressions dict:
                test_expressions_file_path = path.join(dataset_path, 'meta_expressions', 'test', 'meta_expressions.json')
                with open(test_expressions_file_path, 'r') as f:
                    test_expressions_by_video = json.load(f)['videos']
                test_videos = set(test_expressions_by_video.keys())
                valid_plus_test_videos = set(subset_expressions_by_video.keys())
                valid_videos = valid_plus_test_videos - test_videos
                subset_expressions_by_video = {k: subset_expressions_by_video[k] for k in valid_videos}
                assert len(subset_expressions_by_video) == 202, 'error: incorrect number of validation expressions'

                samples_list = []
                for vid_id, data in tqdm(subset_expressions_by_video.items()):
                    vid_frames_indices = sorted(data['frames'])
                    for exp_id, exp_dict in data['expressions'].items():
                        exp_dict['exp_id'] = exp_id
                        samples_list.append((vid_id, vid_frames_indices, exp_dict))

            with open(metadata_file_path, 'w') as f:
                json.dump(samples_list, f)
        if distributed:
            dist.barrier()
            with open(metadata_file_path, 'r') as f:
                samples_list = [tuple(a) for a in tqdm(json.load(f), disable=distributed and dist.get_rank() != 0)]
        return samples_list

    @staticmethod
    def _frame_stride_from_video(all_frame_indices, frame_interval):
        if len(all_frame_indices) < 2:
            return 1
        numeric_indices = [int(frame_idx) for frame_idx in all_frame_indices]
        diffs = [b - a for a, b in zip(numeric_indices[:-1], numeric_indices[1:]) if b > a]
        if not diffs:
            return 1
        base_step = min(diffs)
        return max(1, int(round(frame_interval / max(base_step, 1))))

    def _sample_train_frame_indices(self, video_id, all_frame_indices, obj_id):
        if len(all_frame_indices) <= self.window_size:
            padded = list(all_frame_indices)
            while len(padded) < self.window_size:
                padded.append(padded[-1])
            return padded

        stride = self._frame_stride_from_video(all_frame_indices, self.frame_interval)
        span = stride * (self.window_size - 1)
        if len(all_frame_indices) - 1 < span:
            sampled_positions = np.linspace(0, len(all_frame_indices) - 1, self.window_size)
            sampled_positions = [int(round(pos)) for pos in sampled_positions]
            return [all_frame_indices[pos] for pos in sampled_positions]

        visible_frames = set(self.subset_metas_by_video[video_id]['objects'][obj_id]['frames'])
        visible_positions = [i for i, frame_idx in enumerate(all_frame_indices) if frame_idx in visible_frames]
        if not visible_positions:
            visible_positions = list(range(len(all_frame_indices)))

        anchor_pos = random.choice(visible_positions)
        max_start = len(all_frame_indices) - 1 - span
        candidate_starts = [anchor_pos - slot * stride for slot in range(self.window_size)]
        candidate_starts = [start for start in candidate_starts if 0 <= start <= max_start]
        if not candidate_starts:
            start = min(max(anchor_pos - span // 2, 0), max_start)
        else:
            start = random.choice(candidate_starts)
        return [all_frame_indices[start + slot * stride] for slot in range(self.window_size)]

    @staticmethod
    def bounding_box(img):
        img = img.numpy()
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
    
    def __getitem__(self, idx):
        video_id, frame_indices, text_query_dict = self.samples_list[idx]
        text_query = text_query_dict['exp']
        text_query = " ".join(text_query.lower().split())  # clean up the text query
        if self.subset_type == 'train':
            frame_indices = self._sample_train_frame_indices(video_id, frame_indices, text_query_dict['obj_id'])

        # read the source window frames:
        frame_paths = [path.join(self.videos_dir, video_id, f'{idx}.jpg') for idx in frame_indices]
        source_frames = [Image.open(p) for p in frame_paths]
        original_frame_size = source_frames[0].size[::-1] #[H W]
        h, w = original_frame_size

        if self.subset_type == 'train':
            # read the instance masks:
            annotation_paths = [path.join(self.mask_annotations_dir, video_id, f'{idx}.png') for idx in frame_indices]
            mask_annotations = [torch.tensor(np.array(Image.open(p))) for p in annotation_paths]
            all_object_indices = set().union(*[m.unique().tolist() for m in mask_annotations])
            all_object_indices.remove(0)  # remove the background index
            all_object_indices = sorted(list(all_object_indices))
            mask_annotations_by_object = []
            box_annotations_by_object = []
            for obj_id in all_object_indices:
                frames_mask_annotations = []
                frames_box_annotations = []
                for m in mask_annotations:
                    obj_id_mask_annotation = (m == obj_id).to(torch.uint8)

                    

                    if obj_id_mask_annotation.any() > 0:
                        y1, y2, x1, x2 = self.bounding_box(obj_id_mask_annotation)
                        box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    else:
                        box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    frames_mask_annotations.append(obj_id_mask_annotation)
                    frames_box_annotations.append(box)
                
                obj_id_mask_annotations = torch.stack(frames_mask_annotations)
                obj_id_box_annotations = torch.stack(frames_box_annotations) #[o 4]

                obj_id_box_annotations[:, 0::2].clamp_(min=0, max=w)
                obj_id_box_annotations[:, 1::2].clamp_(min=0, max=h)

                # obj_id_mask_annotations = torch.stack([(m == obj_id).to(torch.uint8) for m in mask_annotations])
                box_annotations_by_object.append(obj_id_box_annotations) 
                mask_annotations_by_object.append(obj_id_mask_annotations)
            mask_annotations_by_object = torch.stack(mask_annotations_by_object)
            box_annotations_by_object = torch.stack(box_annotations_by_object)
            
            mask_annotations_by_frame = rearrange(mask_annotations_by_object, 'o t h w -> t o h w')  # o for object
            box_annotations_by_frame = rearrange(box_annotations_by_object, 'o t c -> t o c') #[object t 4]
            # next we get the referred instance index in the list of all the object ids:
            ref_obj_idx = torch.tensor(all_object_indices.index(int(text_query_dict['obj_id'])), dtype=torch.long)

            category = self.subset_metas_by_video[video_id]['objects'][text_query_dict['obj_id']]['category']

            # create a target dict for each frame:
            targets = []
            for frame_masks, frames_box in zip(mask_annotations_by_frame, box_annotations_by_frame):
                target = {'masks': frame_masks[ref_obj_idx].unsqueeze(0),
                          'boxes': frames_box[ref_obj_idx].unsqueeze(0), #[i 4]
                          # idx in 'masks' of the text referred instance
                          'referred_instance_idx': torch.tensor(0),
                          # whether the referred instance is visible in the frame:
                          'is_ref_inst_visible': frame_masks[ref_obj_idx].any(),
                          'orig_size': frame_masks.shape[-2:],  # original frame shape without any augmentations
                          'labels': torch.tensor([ytvos_category_dict[category]],dtype=torch.long),
                          # size with augmentations, will be changed inside transforms if necessary
                          'size': frame_masks.shape[-2:],
                          'iscrowd': torch.zeros(len(frame_masks)),  # for compatibility with DETR COCO transforms
                          }
                targets.append(target)
        else:
            # validation subset has no annotations, so create dummy targets:
            targets = len(source_frames) * [{
                "size": original_frame_size
            }]

        source_frames, targets, text_query = self.transforms(source_frames, targets, text_query)

        if self.subset_type == 'train':
            return source_frames, targets, text_query
        else:  # validation:
            video_metadata = {'video_id': video_id,
                              'frame_indices': frame_indices,
                              'resized_frame_size': source_frames.shape[-2:],
                              'original_frame_size': original_frame_size,
                              'exp_id': text_query_dict['exp_id']}
            return source_frames, video_metadata, targets, text_query
         
    def __len__(self):
        return len(self.samples_list)


class A2dSentencesTransforms:
    def __init__(self, subset_type, horizontal_flip_augmentations, resize_and_crop_augmentations,
                 random_color, train_short_size, train_max_size, eval_short_size, eval_max_size, **kwargs):
        self.h_flip_augmentation = subset_type == 'train' and horizontal_flip_augmentations
        self.random_color = subset_type == 'train' and random_color
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        scales = [train_short_size]  # no more scales for now due to GPU memory constraints. might be changed later
        self.photometricDistort = T.PhotometricDistort()
        transforms = []
        if resize_and_crop_augmentations:
            if subset_type == 'train':
                transforms.append(T.RandomResize(scales, max_size=train_max_size))
            # elif subset_type == 'test':
            else:
                transforms.append(T.RandomResize([eval_short_size], max_size=eval_max_size)),
        transforms.extend([T.ToTensor(), normalize])
        self.size_transforms = T.Compose(transforms)

    def __call__(self, source_frames, targets, text_query):
        if self.h_flip_augmentation and torch.randn(1) > 0.5:
            source_frames = [F.hflip(f) for f in source_frames]
            for t in targets:
                h, w = t['size']
                t['masks'] = F.hflip(t['masks'])
                boxes = t['boxes'] 
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                t["boxes"] = boxes
            # Note - is it possible for both 'right' and 'left' to appear together in the same query. hence this fix:
            text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
        if self.random_color and torch.randn(1) > 0.5:
            source_frames, targets = self.photometricDistort(source_frames, targets)
        source_frames, targets = list(zip(*[self.size_transforms(f, t) for f, t in zip(source_frames, targets)]))
        source_frames = torch.stack(source_frames)  # [T, 3, H, W]
        return source_frames, targets, text_query


class Collator:
    def __init__(self, subset_type):
        self.subset_type = subset_type

    def __call__(self, batch):
        if self.subset_type == 'train':
            samples, targets, text_queries = list(zip(*batch))
            samples = nested_tensor_from_videos_list(samples)  # [T, B, C, H, W]
            # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
            targets = list(zip(*targets))
            batch_dict = {
                'samples': samples,
                'targets': targets,
                'text_queries': text_queries
            }
            return batch_dict
        else:  # validation:
            samples, videos_metadata, targets, text_queries = list(zip(*batch))
            targets = list(zip(*targets))
            samples = nested_tensor_from_videos_list(samples)  # [T, B, C, H, W]
            batch_dict = {
                'samples': samples,
                'videos_metadata': videos_metadata,
                'text_queries': text_queries,
                'targets': targets
            }
            return batch_dict
