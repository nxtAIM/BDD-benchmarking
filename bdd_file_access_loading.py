import os
import random
import cv2
import time
import argparse

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torchcodec.decoders import SimpleVideoDecoder

from torchvision.transforms import v2

def setup():

    # Initializes a communication group using 'nccl' as the backend for GPU communication.
    dist.init_process_group(backend='nccl')

    world_size = dist.get_world_size()

    # Get the identifier of each process within a node
    local_rank = int(os.getenv('LOCAL_RANK'))

    # Get the global identifier of each process within the distributed system
    rank = int(os.environ['RANK'])

    # Creates a torch.device object that represents the GPU to be used by this process.
    device = torch.device('cuda', local_rank)
    # Sets the default CUDA device for the current process, 
    # ensuring all subsequent CUDA operations are performed on the specified GPU device.
    torch.cuda.set_device(device)

    # Different random seed for each process.
    torch.random.manual_seed(1000 + dist.get_rank())

    return local_rank, rank, device, world_size

class MovDatasetMultiFrame(Dataset):
    def __init__(self, video_dir, annotation_file, num_frames=4, transform=None):

        self.video_dir = video_dir
        self.num_frames = num_frames

        with open(annotation_file, "r") as f:
            self.video_names = f.read().splitlines()

        self.frames = []
        self.lengthsFour = []
        self.total_frames = 0
        
        for video_name in tqdm(self.video_names):
            cap = cv2.VideoCapture(os.path.join(self.video_dir, video_name))
            
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
            
            if total_frames % self.num_frames != 0:
                total_frames = total_frames - (total_frames % self.num_frames)

            self.lengthsFour.append(list(range(0, int(total_frames), self.num_frames)))
            self.frames.append(len(self.lengthsFour[-1])-1)


        self.frames = np.array(self.frames)
        self.frames_cum = np.cumsum(self.frames)
        self.total_frames = int(self.frames_cum[-1])

        self.transform = transform

    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):

        key_index = np.searchsorted(self.frames_cum, idx)

        frame_index = int(abs(self.frames_cum[key_index] - idx))
        frame_index = self.lengthsFour[key_index][frame_index]

        video_name = self.video_names[key_index]
        video = SimpleVideoDecoder(os.path.join(self.video_dir, video_name))

        frames = video[frame_index:frame_index+4]

        if self.transform is not None:
            frames = self.transform(frames)

        return frames
    

def print0(*args):
    if dist.get_rank() == 0:
        print(*args)

def set_seed(seed=1000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--size", type=int, default=224)
    args.add_argument("--video_dir", type=str, default="/p/scratch/genai-ad/benassou1/bdd_videos_filtered") 
    args.add_argument("--video_names", type=str, default="/p/scratch/genai-ad/benassou1/bdd100k_video_h5_names.txt") 
    args = args.parse_args()

    local_rank, rank, device, world_size = setup()

    print0('File system Access ')
    print0(f'with {args}')

    set_seed()
    g = torch.Generator()
    g.manual_seed(0)

    transform = v2.Compose([
            v2.Resize((args.size, args.size), antialias=True),
    ])

    t0 = time.perf_counter()
    dataset = MovDatasetMultiFrame(args.video_dir, args.video_names, transform=transform)
    print0(f"Dataset initialization took: {time.perf_counter()-t0}")
    
    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        seed=1000,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=int(os.getenv('SRUN_CPUS_PER_TASK')),
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print0(f"data {len(dataloader)}")

    t1 = time.perf_counter()
    for data in dataloader:
        data = data.to(device)
        print0(data.shape)      

    print0(f"Loading took: {time.perf_counter()-t1}")
    
