import os
import time
import argparse

import numpy as np

import torch

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import (
    DALIGenericIterator,
    LastBatchPolicy,
)

def setup():
    # Initializes a communication group using 'nccl' as the backend for GPU communication.
    torch.distributed.init_process_group(backend='nccl')
    # Get the identifier of each process within a node
    local_rank = int(os.getenv('LOCAL_RANK'))

    # Get the global identifier of each process within the distributed system
    rank = int(os.environ['RANK'])

    # Creates a torch.device object that represents the GPU to be used by this process.
    device = torch.device('cuda', local_rank)

    world_size = int(os.environ['WORLD_SIZE'])

    # Sets the default CUDA device for the current process, 
    # ensuring all subsequent CUDA operations are performed on the specified GPU device.
    torch.cuda.set_device(device)
    # Different random seed for each process.
    # torch.random.manual_seed(args.seed + torch.distributed.get_rank())

    return rank, local_rank, world_size

def print0(*args):
    if torch.distributed.get_rank() == 0:
        print(*args)

@pipeline_def
def video_pipe(filenames, device="gpu",sequence_length=1, stride=5, shard_id=0, num_shards=1):
    resized_videos = fn.readers.video_resize(
        device=device,
        filenames=filenames,
        sequence_length=sequence_length,
        stride=stride,
        shard_id=shard_id,
        num_shards=num_shards,
        dtype=types.DALIDataType.FLOAT,
        random_shuffle=True,
        skip_vfr_check=True,
        name="Reader",
        lazy_init=False,
        file_list_include_preceding_frame=False,
        pad_last_batch=LastBatchPolicy.PARTIAL,
        prefetch_queue_depth=1,
        stick_to_shard=True,
        resize_x=224, 
        resize_y=224,
    )

    return resized_videos



def main(args):

    shard_id, device_id, num_shards = setup()
    print0('Dali Access ')
    print0(f'with {args} ')
    
    video_directory = args.data_dir
    video_files = [video_directory + "/" + f for f in os.listdir(video_directory)]

    video_pipeline = video_pipe(
        filenames = video_files,
        device="gpu",
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        device_id=device_id,
        shard_id=shard_id,
        num_shards=num_shards,
        num_threads=int(os.getenv('SRUN_CPUS_PER_TASK')),
    )

    start_time = time.time()
    train_loader = DALIGenericIterator(
        video_pipeline,
        output_map=["videos"],
        reader_name="Reader",
        auto_reset=True,
        prepare_first_batch=False,
    )
    
    init_runtime = time.time() -  start_time
    
    print0("Trainloader Init time: ", init_runtime)

    print0("train: ", len(train_loader))
    t0 = time.perf_counter()
    for data in train_loader:
        print0(data[0]['videos'].shape)  

    print0(time.perf_counter()-t0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Example PyTorch Lightning Script for JSC")

    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--data-dir', type=str, default="/p/scratch/genai-ad/benassou1/bdd_videos_filtered", help='Path to the dataset')
    parser.add_argument('--sequence-length', type=int, default=4, help='How many frames per sequence')
    parser.add_argument('--stride', type=int, default=1, help='How many frames to skip between sampling')
    
    # Parse the arguments
    args = parser.parse_args()

    main(args)