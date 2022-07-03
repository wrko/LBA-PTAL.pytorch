from torch.utils import data

from config import hp, path
from model import PHAR
from dataLoader import HARDataSet


model = PHAR()
# model.load(epoch=240)

print("Training data loading...")
train_dataset = HARDataSet(data_path=path.air_train_data, actions=hp.actions)
train_data_loader = data.DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)

print("Test data loading...")
test_dataset = HARDataSet(data_path=path.air_test_data, actions=hp.actions)
test_data_loader = data.DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)

for epoch in range(0, hp.epochs + 1, 1):
    model.train(epoch, train_data_loader, test_data_loader)
