import os
import glob
import pandas as pd

from torch.utils import data

from config import hp, path
from model import PHAR
from dataLoader import HARDataSet
from extractAIR import split_files


# load model
model = PHAR()
# model.load(epoch=240)

# split training and test files
files = glob.glob(os.path.join(path.air_extracted, "*A005*.npz"))
train_files, test_files = split_files(files, 9 / 10, 'subject')

# save file names
df_train = pd.DataFrame(train_files)
df_train.to_csv(os.path.join(path.lstm_model, 'files_for_train.csv'), index=False, header=None)
df_test = pd.DataFrame(test_files)
df_test.to_csv(os.path.join(path.lstm_model, 'files_for_test.csv'), index=False, header=None)

# load training data
print("Training data loading...")
train_dataset = HARDataSet(data_path=path.air_extracted, data_files=train_files)
train_data_loader = data.DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)

# load test data
print("Test data loading...")
test_dataset = HARDataSet(data_path=path.air_extracted, data_files=test_files)
test_data_loader = data.DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)

# train model
for epoch in range(0, hp.epochs + 1, 1):
    model.train(epoch, train_data_loader, test_data_loader)
