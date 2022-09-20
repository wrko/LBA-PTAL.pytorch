import os
import glob

from torch.utils import data

from model import PHAR
from dataLoader import HARDataSet
from config import hp, path

# set False if no GPU is available
hp.use_cuda = True

# load model
epoch = 290
model = PHAR()
model.load(epoch=epoch)

# data setting
hp.b_use_noise = False
hp.actions = ['A005']

# get test data files
test_files = open(os.path.join(path.lstm_model, 'files_for_test.csv'), 'r').read().splitlines()
air_files = glob.glob(os.path.join(path.air_extracted, '*.npz'))

# uncertainty of train dataset
test_dataset = HARDataSet(data_path=path.air_extracted, data_files=test_files, b_pbar=True)
test_data_loader = data.DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=True)
acc, uncertainty = model.test(test_data_loader)
print(f'1: accuracy\t{acc:0.4f}\tuncertainty\t{uncertainty:0.5f}')

# uncertainty of other action (entering room) in test dataset
test_files_to_compare = [file for file in air_files if 'A001' in file]
compare_dataset = HARDataSet(data_path=path.air_extracted, data_files=test_files_to_compare, b_pbar=True)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True)
_, uncertainty = model.test(compare_data_loader)
print(f'2: uncertainty\t{uncertainty:0.5f}')

# uncertainty of other action (standing) in test dataset
test_files_to_compare = [file for file in air_files if 'A004' in file]
compare_dataset = HARDataSet(data_path=path.air_extracted, data_files=test_files_to_compare, b_pbar=True)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True)
acc, uncertainty = model.test(compare_data_loader)
print(f'3: accuracy\t{acc:0.4f}\tuncertainty\t{uncertainty:0.5f}')

# uncertainty of other action (hugging) in test dataset
test_files_to_compare = [file for file in air_files if 'A006' in file]
compare_dataset = HARDataSet(data_path=path.air_extracted, data_files=test_files_to_compare, b_pbar=True)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True)
_, uncertainty = model.test(compare_data_loader)
print(f'4: uncertainty\t{uncertainty:0.5f}')

# uncertainty of same action (handshake) in other dataset
hp.step = 2
compare_dataset = HARDataSet(data_path=path.ntu_extracted, actions=['A058'], b_pbar=True)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True)
acc, uncertainty = model.test(compare_data_loader)
print(f'5: accuracy\t{acc:0.4f}\tuncertainty\t{uncertainty:0.5f}')

# uncertainty of other action (hugging) in other dataset
hp.step = 2
compare_dataset = HARDataSet(data_path=path.ntu_extracted, actions=['A055'], b_pbar=True)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True)
acc, uncertainty = model.test(compare_data_loader)
print(f'6: uncertainty\t{uncertainty:0.5f}')

# uncertainty of other action (kicking) in other dataset
hp.step = 2
compare_dataset = HARDataSet(data_path=path.ntu_extracted, actions=['A051'], b_pbar=True)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True)
acc, uncertainty = model.test(compare_data_loader)
print(f'7: accuracy\t{acc:0.4f}\tuncertainty\t{uncertainty:0.5f}')
