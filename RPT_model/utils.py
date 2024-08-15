import numpy as np
import random
import torch
import os
import h5py
import json
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import IPython
e = IPython.embed
import cv2 as cv

CROP_TOP = True  # hardcode
FILTER_MISTAKES = False  # Filter out mistakes from the dataset even if not use_language

class  EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, policy_class, max_len=None, chunk_size=None, 
                 command_list=None, use_language=False, language_encoder=None, use_gpos=True, use_diff=True, action_is_qpos=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0] 
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        
        self.max_len = max_len
        self.chunk_size = chunk_size if (chunk_size != None) and (chunk_size<max_len) else max_len
        
        self.command_list = [cmd.strip("'\"") for cmd in command_list]
        self.use_language = use_language
        self.language_encoder = language_encoder
        self.transformations = None
        
        self.use_diff = use_diff
        self.action_is_qpos = action_is_qpos
        self.policy_class = policy_class
        if self.policy_class == 'Diffusion':
            self.augment_images = True
        else:
            self.augment_images = False
        
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        max_len = self.max_len

        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        
        if self.use_language or FILTER_MISTAKES:
            
            json_name = f"episode_{episode_id}_encoded_{self.language_encoder}.json"
            encoded_json_path = os.path.join(self.dataset_dir, json_name)
            
            with open(encoded_json_path, "r") as f:
                episode_data = json.load(f)
             
        if self.use_language or FILTER_MISTAKES:
            while True:
                # Randomly sample a segment
                segment = np.random.choice(episode_data) 
                current_idx = episode_data.index(segment)
                if (current_idx + 1 < len(episode_data) and episode_data[current_idx + 1]["type"] == "correction"):
                    continue
                segment_start, segment_end = (segment["start_timestep"],segment["end_timestep"],)
                # if end and start are too close, skip
                if segment_end - segment_start + 1 < 20:
                    continue
                command_embedding = torch.tensor(segment["embedding"]).squeeze()
                break    
        
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            self.is_sim = is_sim
            original_action_shape = root['/action'].shape
            
            if len(self.command_list) > 0 or self.use_language:
                # Sample within the segment boundaries
                start_ts = np.random.randint(segment_start, segment_end) 
                end_ts = min(segment_end, start_ts + max_len - 2)
            else:
                start_ts = np.random.choice(original_action_shape[0])
                end_ts = original_action_shape[0] - 1

            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            gpos = root['/observations/gpos'][start_ts]
            if self.use_diff:
                qpos_diff = [a-b for a,b in zip(qpos, root['/observations/qpos'][0])]
                qpos = np.append(qpos, qpos_diff[:7])
                gpos_diff = [a-b for a,b in zip(gpos, root['/observations/gpos'][0])]
                gpos = np.append(gpos, gpos_diff[:7])
            
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                
            # get all actions after and including start_ts
    
            action_len = end_ts - start_ts + 1 
            if action_len <= self.chunk_size:

                if self.action_is_qpos:
                    action = root['/action_qpos'][start_ts : end_ts + 1]
                else:
                    action = root['/action'][start_ts : end_ts + 1]
                    
            else:
                if self.action_is_qpos:
                    action = root['/action_qpos'][start_ts : start_ts + self.chunk_size]
                else:
                    action = root['/action'][start_ts : start_ts + self.chunk_size]
            action_len = min(action_len, self.chunk_size)    
            
            # add history image and action 
            history_images = []
            history_image_dict = dict()

            qpos_his = root['/observations/qpos'][0:start_ts + 1] 
            qpos_his_len = len(qpos_his) 

            gripper_state = qpos_his[qpos_his_len-1][7] 
            gripper_change_point = []

            for idx in range(qpos_his_len-1, -1, -1): 
                if gripper_state != qpos_his[idx][7]:
                    gripper_state = qpos_his[idx][7]
                    gripper_change_point.append(idx+1)
                    break
            gripper_change_point.append(0)

            change_point = gripper_change_point[0] 
            history_action_len = start_ts - change_point 

            if history_action_len<=(self.chunk_size): 
                
                history_action = root['/action'][change_point : start_ts] # 10,15

                for history_idx in range(change_point, start_ts):
                    for cam_name in self.camera_names:
                        history_image_dict[cam_name] = root[f'/observations/images/{cam_name}'][history_idx]
                    history_images.append(history_image_dict.copy())
            else:
                history_action = root['/action'][start_ts - self.chunk_size : start_ts] # chunk 10,change 10, start40,40 - 10 = 30, 30 ~ 10
                for history_idx in range(start_ts - self.chunk_size, start_ts):
                    for cam_name in self.camera_names:
                        history_image_dict[cam_name] = root[f'/observations/images/{cam_name}'][history_idx]
                    history_images.append(history_image_dict.copy())
            history_action_len = min(history_action_len, self.chunk_size)

            # use diff
            if self.use_diff:
                qpos = root['/observations/qpos'][start_ts]
                qpos_diff = [a-b for a,b in zip(qpos, root['/observations/qpos'][change_point])]
                qpos = np.append(qpos, qpos_diff[:7])
                
                gpos = root['/observations/gpos'][start_ts]
                gpos_diff = [a-b for a,b in zip(gpos, root['/observations/gpos'][change_point])]
                gpos = np.append(gpos, gpos_diff[:7])
                         
            # else: 
            #     history_action_len = start_ts + 1
            #     if history_action_len <= self.chunk_size: 
            #         history_action = root['/action'][0 : start_ts + 1]
            #         for history_idx in range(start_ts + 1):
            #             for cam_name in self.camera_names:
            #                 history_image_dict[cam_name] = root[f'/observations/images/{cam_name}'][history_idx]
            #             history_images.append(history_image_dict.copy())
            #     else:
            #         history_action = root['/action'][start_ts - self.chunk_size : start_ts]
            #         for history_idx in range(start_ts - self.chunk_size, start_ts + 1): 
            #             for cam_name in self.camera_names:
            #                 history_image_dict[cam_name] = root[f'/observations/images/{cam_name}'][history_idx]
            #             history_images.append(history_image_dict.copy())
            #     history_action_len = min(history_action_len, self.chunk_size)
            
            
        padded_action = np.zeros((self.chunk_size,) + original_action_shape[1:], dtype=np.float32) 
        padded_action[:action_len] = action[:action_len]
        padded_history_action = np.zeros((self.chunk_size,) + original_action_shape[1:], dtype=np.float32)
        is_pad_action = np.zeros(self.chunk_size)
        is_pad_action[action_len:] = 1
        
        history_action = history_action[::-1] 
        padded_history_action[:history_action_len] = history_action[:history_action_len] 
        is_pad_history = np.zeros(self.chunk_size)
        is_pad_history[history_action_len:] = 1 
        
        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            if "ACT" in self.policy_class: # and self.policy_class!= "ACT0E0":
                image_dict[cam_name] = cv.resize(image_dict[cam_name], (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
                
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        
        # history_images 
        history_all_cam_images = []
        for history_idx in range(history_action_len):
            all_history_cam_images = []
            for cam_name in self.camera_names: 
                if "ACT" in self.policy_class: # and self.policy_class!= "ACT0E0":
                    history_images[history_idx][cam_name] = cv.resize(history_images[history_idx][cam_name], (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
                    
                all_history_cam_images.append(history_images[history_idx][cam_name])
            all_history_cam_images = np.stack(all_cam_images, axis=0)
            history_all_cam_images.append(all_history_cam_images)
        
        original_images_shape = np.shape(all_cam_images) # np.shape(history_all_cam_images) 
        padded_history_images = np.zeros((self.chunk_size,) + original_images_shape[:], dtype=np.float32 )# just for 1 camera now
        history_all_cam_images = history_all_cam_images[::-1]  # reverse
        if history_all_cam_images != []:
            padded_history_images[:history_action_len] = history_all_cam_images[:history_action_len] # padded_history_images
        
        history_all_cam_images = np.array(padded_history_images)
        history_images_data = torch.from_numpy(history_all_cam_images)
        # Constructing the observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        gpos_data = torch.from_numpy(gpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad_action = torch.from_numpy(is_pad_action).bool()
        history_action_data = torch.from_numpy(padded_history_action).float()
        
        is_pad_history = torch.from_numpy(is_pad_history).bool()
        
        
        # Adjusting channel
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        
        history_images_data = torch.einsum('q k h w c -> q k c h w', history_images_data) 
        
        # normalize image and change dtype to float
        image_data = image_data / 255.0
        history_images_data = history_images_data / 255.0
        
        # np.set_printoptions(linewidth=300)
        if self.policy_class == 'Diffusion':

            if self.transformations is None:
                print('Initializing transformations')
                print(f'{self.augment_images=}')
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
                ]
            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)
            # normalize to [-1, 1]
            action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
        else:
            # normalize to mean 0 std 1
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        gpos_data = (gpos_data - self.norm_stats["gpos_mean"]) / self.norm_stats["gpos_std"]
        
        history_action_data = (history_action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        
        # return
        if self.use_language:
            return image_data, qpos_data, gpos_data, history_images_data, history_action_data, is_pad_history, action_data, is_pad_action, command_embedding
        else:
            return image_data, qpos_data, gpos_data, history_images_data,  history_action_data, is_pad_history, action_data, is_pad_action

def get_norm_stats(dataset_dir, num_episodes, use_gpos=True, use_diff=True, action_is_qpos=False):
    all_qpos_data = []
    all_gpos_data = []
    all_action_data = []
    max_episode_len = 0    
            
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            episode_len = root['/action'].shape[0]
            qpos = root['/observations/qpos'][()]
            gpos = root['/observations/gpos'][()]
            if action_is_qpos:
                action = root['/action_qpos'][()]
            else:
                action = root['/action'][()]
                
        if use_diff: 
            qpos_len = len(qpos) 
            gripper_state = qpos[qpos_len-1][7]
            gripper_change_point = []

            for idx in range(qpos_len-1, -1, -1):
                if gripper_state != qpos[idx][7]:
                    gripper_state = qpos[idx][7]
                    gripper_change_point.append(idx+1)
                    # break
            gripper_change_point.append(0)
            
            for idx in range(qpos.shape[0]):
                for c_idx in range(len(gripper_change_point)):
                    if gripper_change_point[c_idx] <= idx: 
                        last_change_point = gripper_change_point[c_idx]
                        break
                # print(f"{idx=}, {last_change_point=}")
                qpos_diff = [a-b for a,b in zip(qpos[idx], qpos[last_change_point])] 
                gpos_diff = [a-b for a,b in zip(gpos[idx], gpos[last_change_point])]
                
                qpos_temp = np.concatenate((qpos[idx],qpos_diff[:7]), axis=0)[np.newaxis,:]
                gpos_temp = np.concatenate((gpos[idx],gpos_diff[:7]), axis=0)[np.newaxis,:]
                
                qpos_data = qpos_temp if idx == 0 else np.concatenate((qpos_data, qpos_temp), axis=0)   
                gpos_data = gpos_temp if idx == 0 else np.concatenate((gpos_data, gpos_temp), axis=0)           

            all_qpos_data = qpos_data if episode_idx == 0 else np.concatenate((all_qpos_data, qpos_data), axis=0)
            all_gpos_data = gpos_data if episode_idx == 0 else np.concatenate((all_gpos_data, gpos_data), axis=0)
            all_action_data = action if episode_idx == 0 else np.concatenate((all_action_data, action), axis=0)
        else: 
            all_qpos_data = qpos if episode_idx == 0 else np.concatenate((all_qpos_data, qpos), axis=0)
            all_gpos_data = gpos if episode_idx == 0 else np.concatenate((all_gpos_data, gpos), axis=0)
            all_action_data = action if episode_idx == 0 else np.concatenate((all_action_data, action), axis=0)   

    all_qpos_data = np.array(all_qpos_data)
    all_gpos_data = np.array(all_gpos_data)
    all_action_data = np.array(all_action_data)
    
    all_qpos_data = torch.from_numpy(all_qpos_data) 
    all_gpos_data = torch.from_numpy(all_gpos_data)
    all_action_data = torch.from_numpy(all_action_data) 

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True).unsqueeze(0) # [1, 1, 8]
    action_std = all_action_data.std(dim=0, keepdim=True).unsqueeze(0)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping
    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    
    # normalize qpos data
    if use_diff: 
        qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True).unsqueeze(0) # [1, 1, 1] 
        qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True).unsqueeze(0)
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
        
        gpos_mean = all_gpos_data.mean(dim=[0, 1], keepdim=True).unsqueeze(0) # [1, 1, 1]
        gpos_std = all_gpos_data.std(dim=[0, 1], keepdim=True).unsqueeze(0)
        gpos_std = torch.clip(gpos_std, 1e-2, np.inf) # clipping
    else:
        qpos_mean = all_qpos_data.mean(dim=0, keepdim=True).unsqueeze(0) # [1, 1, 8]  
        qpos_std = all_qpos_data.std(dim=0, keepdim=True).unsqueeze(0)
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
        
        gpos_mean = all_gpos_data.mean(dim=0, keepdim=True).unsqueeze(0) # [1, 1, 8]
        gpos_std = all_gpos_data.std(dim=0, keepdim=True).unsqueeze(0)
        gpos_std = torch.clip(gpos_std, 1e-2, np.inf) # clipping
    
    eps = 0.0001
    stats = {"action_mean": action_mean.numpy().squeeze(), 
             "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), 
             "qpos_std": qpos_std.numpy().squeeze(),
             "gpos_mean": gpos_mean.numpy().squeeze(), 
             "gpos_std": gpos_std.numpy().squeeze(),
             "example_qpos": qpos,
             "example_gpos": gpos,
             "action_min": action_min.numpy() - eps,
             "action_max": action_max.numpy() + eps,
             } 

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, policy_class, max_len=None, chunk_size=None, command_list=None, use_language=False, language_encoder=None, use_gpos=True, use_diff=True, action_is_qpos=False):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_gpos=use_gpos, use_diff=use_diff, action_is_qpos=action_is_qpos)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, policy_class, max_len, chunk_size, command_list, use_language, language_encoder,use_gpos=use_gpos, use_diff=use_diff, action_is_qpos=action_is_qpos)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, policy_class, max_len, chunk_size, command_list, use_language, language_encoder, use_gpos=use_gpos, use_diff=use_diff, action_is_qpos=action_is_qpos)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

### helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
def number_to_one_hot(number, size=501):
    one_hot_array = np.zeros(size)
    one_hot_array[number] = 1
    return one_hot_array

# audio2text parrot
class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_gpu_mem_info(gpu_id=0):

    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} the gpu is not exist!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free

def get_cpu_mem_info():
    import psutil
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used
