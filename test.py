from torch.utils import data

from model import PHAR
from dataLoader import HARDataSet
from config import hp, path

# load model
epoch = 90
hp.use_cuda = False
model = PHAR()
model.load(epoch=epoch)

# uncertainty of train dataset
test_dataset = HARDataSet(data_path=path.air_test_data, actions=hp.actions, b_pbar=False)
test_data_loader = data.DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)
acc, uncertainty = model.test(epoch, test_data_loader)
print(f'1: accuracy\t{acc:0.4f}\tuncertainty\t{uncertainty:0.4f}')

# uncertainty of other action in test dataset
compare_dataset = HARDataSet(data_path=path.air_test_data, actions=['A001'], b_pbar=False)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)
acc, uncertainty = model.test(epoch, compare_data_loader)
print(f'2: accuracy\t{acc:0.4f}\tuncertainty\t{uncertainty:0.4f}')

# uncertainty of other action in test dataset
compare_dataset = HARDataSet(data_path=path.air_test_data, actions=['A004'], b_pbar=False)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)
acc, uncertainty = model.test(epoch, compare_data_loader)
print(f'3: accuracy\t{acc:0.4f}\tuncertainty\t{uncertainty:0.4f}')

# uncertainty of other action in test dataset
compare_dataset = HARDataSet(data_path=path.air_test_data, actions=['A006'], b_pbar=False)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)
acc, uncertainty = model.test(epoch, compare_data_loader)
print(f'4: accuracy\t{acc:0.4f}\tuncertainty\t{uncertainty:0.4f}')

# uncertainty of same action (handshake) in other dataset
hp.step = 2
compare_dataset = HARDataSet(data_path=path.ntu_test_data, actions=['A058'], b_pbar=False)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)
acc, uncertainty = model.test(epoch, compare_data_loader)
print(f'5: accuracy\tunknown\tuncertainty\t{uncertainty:0.4f}')

# uncertainty of other action (hugging) in other dataset
hp.step = 2
compare_dataset = HARDataSet(data_path=path.ntu_test_data, actions=['A055'], b_pbar=False)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)
acc, uncertainty = model.test(epoch, compare_data_loader)
print(f'6: accuracy\tunknown\tuncertainty\t{uncertainty:0.4f}')

# uncertainty of other action (kicking) in other dataset
hp.step = 2
compare_dataset = HARDataSet(data_path=path.ntu_test_data, actions=['A051'], b_pbar=False)
compare_data_loader = data.DataLoader(compare_dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)
acc, uncertainty = model.test(epoch, compare_data_loader)
print(f'7: accuracy\tunknown\tuncertainty\t{uncertainty:0.4f}')
