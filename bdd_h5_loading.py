import os
import random
import h5py
import time
import argparse
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torchvision.transforms import v2

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def print0(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)

def setup():

    # Initializes a communication group using 'nccl' as the backend for GPU communication.
    torch.distributed.init_process_group(backend='nccl')


    world_size = torch.distributed.get_world_size()
    print(f'world_size: {world_size}')
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
    torch.random.manual_seed(1000 + torch.distributed.get_rank())

    return local_rank, rank, device

class MultiHDF5DatasetMultiFrame(Dataset):
    def __init__(self, hdf5_paths_file, num_frames=4, transform=None):
        self.num_frames = num_frames
        
        with open(os.path.expandvars(hdf5_paths_file), 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.files = [h5py.File(path, 'r') for path in self.hdf5_paths]
        self.lengths = []
        self.keys = []
        self.lengthsFour = []

        for i, file in enumerate(self.files):
            keys = list(file.keys())

            for key in keys:
                frame_length = len(file[key])
                
                if frame_length % self.num_frames != 0:
                    frame_length = frame_length - (frame_length % self.num_frames)
                
                self.lengthsFour.append(list(range(0, frame_length, self.num_frames)))
                self.lengths.append(len(self.lengthsFour[-1])-1)
                self.keys.append([key, i])

        self.lengths = np.array(self.lengths)
        self.lengths_cum = np.cumsum(self.lengths)
        self.total_length = self.lengths_cum[-1]        
        
        self.transform = transform

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        
        key_index = np.searchsorted(self.lengths_cum, idx)

        file_key, file_idx = self.keys[key_index]

        frame_index = abs(self.lengths_cum[key_index] - idx)       
        frame_index = self.lengthsFour[key_index][frame_index]
        
        images = torch.from_numpy(self.files[file_idx][file_key][frame_index:frame_index+self.num_frames])
        images = images.permute(0, 3, 1, 2)

        if self.transform is not None:
            images = self.transform(images)

        return images
    
    def close(self):
        for file in self.files:
            file.close()


def set_seed(seed=1000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--size", type=int, default=224)
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--h5_dir", type=str, default="/p/scratch/genai-ad/mittal3/datasets/bdd100k_h5/videos/train/day_clear/train_bdd_day_clear_2k_videos.txt")
    args = args.parse_args()

    local_rank, rank, device = setup()
    
    print0('H5 files access ')
    print0(f'with {args}')

    set_seed()
    g = torch.Generator()
    g.manual_seed(0)

    transform = v2.Compose([
        v2.Resize((args.size, args.size), antialias=True),
    ])

    t0 = time.perf_counter()
    dataset = MultiHDF5DatasetMultiFrame(hdf5_paths_file=args.h5_dir, transform=transform)
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
    dataset.close()