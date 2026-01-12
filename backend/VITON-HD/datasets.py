import json
from os import path as osp

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms


class VITONDataset(data.Dataset):
    def __init__(self, opt):
        super(VITONDataset, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataset_dir, opt.dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load data list
        img_names = []
        c_names = []
        with open(osp.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)
                c_names.append(c_name)

        self.img_names = img_names
        self.c_names = dict()
        self.c_names['unpaired'] = c_names

    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        # ONLY erase clothes (5: Upper, 6: Dress, 7: Coat)
        # FIX: Also erase Arms (14, 15) to allow checking for long sleeves
        mask_clothes = ((parse_array == 5).astype(np.float32) +
                        (parse_array == 6).astype(np.float32) +
                        (parse_array == 7).astype(np.float32) +
                        (parse_array == 14).astype(np.float32) +
                        (parse_array == 15).astype(np.float32))
        
        # FIX: Protect Hands using Pose
        mask_hands = self.get_hand_mask(pose_data, parse.size)
        mask_clothes = mask_clothes * (1 - mask_hands)

        import cv2
        mask_clothes = cv2.dilate(mask_clothes, np.ones((5, 5), np.uint8), iterations=1)
        
        agnostic_array = parse_array.copy()
        agnostic_array[mask_clothes > 0] = 0
        return Image.fromarray(agnostic_array)

    def get_hand_mask(self, pose_data, shape):
        # FIX: The original logic drew circles around wrists which "cut" the long sleeves.
        # We now return an empty mask (zeros) so the cloth is allowed to cover the wrists/arms.
        # If needed, a better semantic segmentation model should be used instead of this heuristic.
        mask = Image.new('L', shape, 0)
        return np.array(mask).astype(np.float32) / 255.0

    def get_img_agnostic(self, img, parse, pose_data):
        parse_array = np.array(parse)
        # 5: Upper, 6: Dress, 7: Coat
        # FIX: Include 14 (Left Arm) and 15 (Right Arm) to allow Long Sleeves to cover them!
        mask_clothes = ((parse_array == 5).astype(np.float32) +
                        (parse_array == 6).astype(np.float32) +
                        (parse_array == 7).astype(np.float32) +
                        (parse_array == 14).astype(np.float32) +
                        (parse_array == 15).astype(np.float32))
        
        # --- "MULTISPECTRAL" PROTECTION LAYERS ---
        # FIX: Use Pose-based Hand Mask instead of parsing labels 14/15
        mask_hands = self.get_hand_mask(pose_data, img.size)

        import cv2
        # Dilate clothes to cover logos
        mask_clothes = cv2.dilate(mask_clothes, np.ones((10, 10), np.uint8), iterations=1)
        
        # ...Subtract the REAL hands from the clothes mask
        mask_clothes = mask_clothes * (1 - mask_hands)

        img_np = np.array(img).copy()
        
        # BLUR-AGNOSTIC: Blur the clothing area
        shirt_area = img_np.copy()
        shirt_area = cv2.GaussianBlur(shirt_area, (51, 51), 0)
        
        agnostic = img_np.copy()
        mask_bool = mask_clothes > 0
        
        # Fill the cloth area with gray
        agnostic[mask_bool] = [128, 128, 128]

        return Image.fromarray(agnostic)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        c_name = {}
        c = {}
        cm = {}
        for key in self.c_names:
            c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
            c[key] = transforms.Resize(self.load_width, interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            cm[key] = transforms.Resize(self.load_width, interpolation=0)(cm[key])

            c[key] = self.transform(c[key])  # [-1,1]
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)

        # load pose image
        pose_name = img_name.replace('.jpg', '_rendered.png')
        pose_rgb = Image.open(osp.join(self.data_path, 'openpose_img', pose_name))
        pose_rgb = transforms.Resize(self.load_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        pose_name = img_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        # load parsing image
        parse_name = img_name.replace('.jpg', '.png')
        parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        parse = transforms.Resize(self.load_width, interpolation=0)(parse)
        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long() # [1, H, W]
        
        # FIX: Also keep the Original Parse (Unaltered) for "Multispectral" GMM Targeting
        parse_tensor = torch.from_numpy(np.array(parse)[None]).long() # [1, H, W]

        # Create One-Hot Map for Original Parse
        parse_orig_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_orig_map.scatter_(0, parse_tensor, 1.0)
        
        labels = {
            0: ['background', [0]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['neck', [10, 3, 11]]  # Neck and Noise
        }
        parse_agnostic_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # load person image
        img_orig = Image.open(osp.join(self.data_path, 'image', img_name)).convert('RGB')
        img = transforms.Resize(self.load_width, interpolation=2)(img_orig)
        img_agnostic = self.get_img_agnostic(img, parse, pose_data)
        img = self.transform(img)
        img_agnostic = self.transform(img_agnostic)  # [-1,1]
        
        # Calculate Hand Mask for final composition
        hand_mask_np = self.get_hand_mask(pose_data, img_orig.size) # [0,1] float
        hand_mask_tensor = torch.from_numpy(hand_mask_np).float().unsqueeze(0) # [1, H, W]

        # Original high-res image for restoration
        img_orig_tensor = transforms.ToTensor()(img_orig) # [0, 1] format for easier blending later

        result = {
            'img_name': img_name,
            'c_name': c_name,
            'img': img,
            'img_orig': img_orig_tensor,
            'img_agnostic': img_agnostic,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_rgb,
            'pose_data': torch.from_numpy(pose_data).float(),
            'parse': parse_agnostic_map,
            'parse_orig': parse_orig_map, # Raw "Blue+Brown" layers
            'cloth': c,
            'cloth_mask': cm,
            'hand_mask': hand_mask_tensor
        }
        return result

    def __len__(self):
        return len(self.img_names)


class VITONDataLoader:
    def __init__(self, opt, dataset):
        super(VITONDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
