import math
import random

import numpy as np

import secretflow as sf

from secretflow.security.aggregation import PlainAggregator
from ml.dataloaders.livertumor import LiverTumor, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Function to create dataset builder
def create_dataset_builderVNet(batch_size=2, random_seed=1234):
    def dataset_builder(base_dir, stage="train"):
        db_train = LiverTumor(
            base_dir=base_dir,
            split='train',
            transform=transforms.Compose([
                RandomRotFlip(),
                RandomCrop(),
                ToTensor(),
            ])
        )

        def worker_init_fn(worker_id):
            random.seed(random_seed + worker_id)

        dataset_size = len(db_train)
        indices = list(range(dataset_size))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        split = int(np.floor(0.8 * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        trainloader = DataLoader(
            db_train,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            sampler=train_sampler
        )
        validloader = DataLoader(
            db_train,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            sampler=valid_sampler
        )

        if stage == "train":
            train_step_per_epoch = math.ceil(split / batch_size)
            return trainloader, train_step_per_epoch
        elif stage == "eval":
            eval_step_per_epoch = math.ceil((dataset_size - split) / batch_size)
            return validloader, eval_step_per_epoch

    return dataset_builder

def initialize_secretflow():
    sf.shutdown()
    sf.init(
        parties=['alice', 'bob', 'carol', 'dave'],
        address='local',
        log_to_driver=False,
        debug_mode=True,
        num_gpus=2
    )

def create_devices():
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    carol = sf.PYU('carol')
    dave = sf.PYU('dave') # server
    return alice, bob, carol, dave, [alice, bob, carol], PlainAggregator(device=dave)