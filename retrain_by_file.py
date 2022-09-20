import os
import glob
import random

from torch.utils import data

from model import PHAR
from dataLoader import HARDataSet
from config import hp, path

# setting
hp.use_cuda = True
hp.learning_rate = 0.0001
epoch = 290
data_path = path.ntu_extracted
n_partition = 10


def retrain(model, train_files, test_files):
    retrain_files = list()
    for idx in range(n_partition - 1):
        partition_size = int(len(train_files) / (n_partition - 1))
        retrain_files.append(train_files[(idx * partition_size):((idx + 1) * partition_size)])

    # load test dataset
    hp.b_use_noise = False
    test_dataset = HARDataSet(data_path=data_path, data_files=test_files, b_pbar=False)
    test_data_loader = data.DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=True)

    # load retrain dataset
    hp.b_use_noise = True
    retrain_data_loaders = list()
    for idx in range(n_partition - 1):
        retrain_dataset = HARDataSet(data_path=data_path, data_files=retrain_files[idx], b_pbar=False)
        retrain_data_loaders.append(
            data.DataLoader(retrain_dataset, batch_size=hp.batch_size, shuffle=True))

    # without retraining
    print('1. Testing without retraining')
    acc, uncertainty = model.test(test_data_loader)
    print(f'Accuracy\t{acc:0.4f}\tUncertainty\t{uncertainty:0.5f}')
    print()

    # retraining using 10% of random data of new subjects
    print('2. Retraining with additional 10% data of new subjects.')
    max_epochs = 1
    for idx in range(n_partition - 1):
        for epoch in range(0, max_epochs + 1, 1):
            model.train(epoch, retrain_data_loaders[idx], test_data_loader, b_print=False)
        acc, uncertainty = model.test(test_data_loader)
        print(f'Accuracy\t{acc:0.4f}\tUncertainty\t{uncertainty:0.5f}')


# load all file names
files = glob.glob(os.path.join(path.ntu_extracted, f"*A058*.npz"))

# load models
model1 = PHAR()
model1.load(epoch=epoch)
model2 = PHAR()
model2.load(epoch=epoch)
model3 = PHAR()
model3.load(epoch=epoch)

# retraining test
print('\n=== Random Split ===\n')
sorted_files = list(files)
random.shuffle(sorted_files)
idx_split = int(len(sorted_files) * (n_partition - 1) / n_partition)
train_files = sorted_files[:idx_split]
test_files = sorted_files[idx_split:]
retrain(model1, train_files, test_files)

# calculate uncertainties
print('\nCalculating uncertainty of each file...\n')
uncertainties = list()
for file in train_files:
    file_dataset = HARDataSet(data_path=path.ntu_extracted, data_files=[file], b_pbar=False)
    file_data_loader = data.DataLoader(file_dataset, batch_size=hp.batch_size, shuffle=True)
    _, uncertainty = model2.test(file_data_loader)
    uncertainties.append(uncertainty)

# retraining test
print('\n\n=== High Uncertain Data First ===\n')
sorted_files = [file for _, file in sorted(zip(uncertainties, train_files), reverse=True)]
retrain(model2, sorted_files, test_files)

# retraining test
print('\n\n=== Low Uncertainty Data First ===\n')
sorted_files = [file for _, file in sorted(zip(uncertainties, train_files), reverse=False)]
retrain(model3, sorted_files, test_files)
