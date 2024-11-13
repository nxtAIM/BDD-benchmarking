import os
import time
import random
import argparse
import pyarrow as pa
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torchvision import transforms
from torchvision import tv_tensors
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

def print0(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)
        

class ArrowDatasetMultiFrame(Dataset):

    def __init__(self, arrow_files, num_frames=4, transform=None):
        self.arrow_files = arrow_files
        self.num_frames = num_frames
        
        self._len = []
        self.lengths = []
        self.keys = []
        self.lengthsFour = []

        for i, file in enumerate(os.listdir(self.arrow_files)):
            with pa.OSFile(os.path.join(self.arrow_files, file), 'rb') as f:
                reader = pa.ipc.open_file(f) 
                for j in range(reader.num_record_batches):
                    batch = reader.get_batch(j)
                    frame_length = batch['shape'][0].as_py()[0] 
                    
                    if frame_length % self.num_frames != 0:
                        frame_length = frame_length - (frame_length % self.num_frames)
                
                    self.lengthsFour.append(list(range(0, frame_length, self.num_frames)))
                    self._len.append(len(self.lengthsFour[-1])-1)
                    self.keys.append([j, i])
        
        self.readers = [pa.ipc.open_file(pa.OSFile(os.path.join(self.arrow_files, file), 'rb')) for file in os.listdir(self.arrow_files)]
        self._len = np.array(self._len)
        self.lengths_cum = np.cumsum(self._len)

        self.total_length = self.lengths_cum[-1] 

        self.transform = transform


    def __len__(self):
        return self.total_length


    def __getitem__(self, idx):
        
        key_index = np.searchsorted(self.lengths_cum, idx)

        file_key, file_idx = self.keys[key_index]
        
        frame_index = abs(self.lengths_cum[key_index] - idx)       
        frame_index = self.lengthsFour[key_index][frame_index]
        # print(f"key_index, frame_index {key_index, frame_index}")
        row = self.readers[file_idx].get_batch(file_key)
        row_shapes = row['shape'][0].as_py()

        array_from_binary = np.frombuffer(row['frames'][0].as_py(),  dtype=np.uint8).reshape(row_shapes)

        images = torch.tensor(array_from_binary[frame_index:frame_index+self.num_frames] )
        images = images.permute(0, 3, 1, 2)

        if self.transform is not None:
            images = self.transform(images)

        return images

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
    args.add_argument("--size", type=int, default=224)
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--pyarrow_dir", type=str, default="/p/scratch/genai-ad/benassou1/bdd_arrows/byte_bdd")
    args = args.parse_args()
    
    local_rank, rank, device, world_size = setup()
    
    print0('PyArrow files access ')
    print0(f'with {args}')

    set_seed()
    g = torch.Generator()
    g.manual_seed(0)

    transform = v2.Compose([
        v2.Resize((args.size, args.size), antialias=True),
    ])

    t0 = time.perf_counter()
    dataset = ArrowDatasetMultiFrame(arrow_files=args.pyarrow_dir, transform=transform)
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
        shuffle=False,
        sampler=sampler,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print0(f"data {len(dataloader)}")
    t1 = time.perf_counter()
    for data in dataloader:
        data = data.to(device)
        print(data.shape)

    print0(f"Loading took: {time.perf_counter()-t1}")
    
