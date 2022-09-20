import os
import glob
import random
from tqdm import tqdm

from torch.utils import data

from model import PHAR
from dataLoader import HARDataSet
from config import hp, path

# setting
hp.use_cuda = True
hp.learning_rate = 0.0005
epoch = 290
data_path = path.ntu_extracted
n_partitions = 30
n_test_partitions = 10
b_use_noise = False


def retrain(phar_model, data_loader):
    # load retrain and test data
    inp_sets, out_sets = list(), list()
    for inps, outs in data_loader:
        inp_sets.append(inps)
        out_sets.append(outs)

    # without retraining
    accuracies, uncertainties = list(), list()
    # print('1. Testing without retraining...')
    acc, uncertainty = test_all(phar_model, inp_sets[-n_test_partitions:], out_sets[-n_test_partitions:])
    accuracies.append(acc)
    uncertainties.append(uncertainty)
    # print(f'Accuracy: {acc:0.4f}, Uncertainty: {uncertainty:0.5f}')
    # print()

    # retraining using 10% of random data of new subjects
    # print('2. Retraining with each data of new subjects...')
    for idx in range(n_partitions - n_test_partitions):
        phar_model.step(inp_sets[idx], out_sets[idx], b_train=True, n_samples=100)
        acc, uncertainty = test_all(phar_model, inp_sets[-n_test_partitions:], out_sets[-n_test_partitions:])
        accuracies.append(acc)
        uncertainties.append(uncertainty)
        # print(f'Accuracy: {acc:0.4f}, Uncertainty: {uncertainty:0.5f}')
    return accuracies, uncertainties


def test_all(phar_model, inp_sets, out_sets):
    n_sets = len(inp_sets)
    sum_acc, sum_uncertainty = 0., 0.
    for test_idx in range(n_sets):
        _, acc, uncertainty = phar_model.step(inp_sets[-test_idx], out_sets[-test_idx], b_train=False, n_samples=100)
        sum_acc += acc
        sum_uncertainty += uncertainty
    return sum_acc / n_sets, sum_uncertainty / n_sets


# load all file names
subject_names = list()
files = glob.glob(os.path.join(path.ntu_extracted, f"*A058*.npz"))
for file in files:
    subject_name = os.path.basename(file)[8:12]
    if subject_name not in subject_names:
        subject_names.append(subject_name)

# for each subject
pbar = tqdm(total=len(subject_names))
n_results = n_partitions - n_test_partitions + 1
mean_accuracies_1, mean_uncertainties_1 = [0.] * n_results, [0.] * n_results
mean_accuracies_2, mean_uncertainties_2 = [0.] * n_results, [0.] * n_results
mean_accuracies_3, mean_uncertainties_3 = [0.] * n_results, [0.] * n_results
for subject_name in subject_names:
    # load models
    phar_model_1 = PHAR()
    phar_model_1.load(epoch=epoch)
    phar_model_2 = PHAR()
    phar_model_2.load(epoch=epoch)
    phar_model_3 = PHAR()
    phar_model_3.load(epoch=epoch)

    # load data of the subject
    subject_files = glob.glob(os.path.join(path.ntu_extracted, f"*{subject_name}*A058*.npz"))
    subject_dataset = HARDataSet(data_path=path.ntu_extracted, data_files=subject_files, b_pbar=False)

    # random shuffle
    inps_and_outs = list(zip(subject_dataset.inps, subject_dataset.outs))
    random.shuffle(inps_and_outs)
    subject_dataset.inps, subject_dataset.outs = zip(*inps_and_outs)
    subject_dataset.inps, subject_dataset.outs = list(subject_dataset.inps), list(subject_dataset.outs)
    partition_size = int(len(subject_dataset.inps) / n_partitions)
    subject_dataset.inps = list(subject_dataset.inps[:partition_size * n_partitions])
    subject_dataset.outs = list(subject_dataset.outs[:partition_size * n_partitions])

    # retraining test 1: random split
    # print('\n=== Random Split ===\n')
    subject_data_loader = data.DataLoader(subject_dataset, batch_size=partition_size, shuffle=False, drop_last=True)
    accuracies, uncertainties = retrain(phar_model_1, subject_data_loader)
    mean_accuracies_1 = [sum(v) for v in zip(mean_accuracies_1, accuracies)]
    mean_uncertainties_1 = [sum(v) for v in zip(mean_uncertainties_1, uncertainties)]

    # calculate uncertainties
    # print('\nCalculating uncertainty of each data...\n')
    data_uncertainties = list()
    for inps, outs in subject_data_loader:
        _, _, values = phar_model_2.step(inps, outs, b_train=False, n_samples=100, b_individual=True)
        data_uncertainties.extend(values)
        if len(data_uncertainties) == partition_size * (n_partitions - n_test_partitions):
            break

    # retraining test 2: high uncertain data first
    # print('\n\n=== High Uncertain Data First ===\n')
    sorted_idxs = sorted(range(len(data_uncertainties)), key=lambda k: data_uncertainties[k], reverse=True)
    temp_inps, temp_outs = list(subject_dataset.inps), list(subject_dataset.outs)
    for cur_idx, sorted_idx in enumerate(sorted_idxs):
        subject_dataset.inps[cur_idx] = temp_inps[sorted_idx]
        subject_dataset.outs[cur_idx] = temp_outs[sorted_idx]
    subject_data_loader = data.DataLoader(subject_dataset, batch_size=partition_size, shuffle=False, drop_last=True)
    accuracies, uncertainties = retrain(phar_model_2, subject_data_loader)
    mean_accuracies_2 = [sum(v) for v in zip(mean_accuracies_2, accuracies)]
    mean_uncertainties_2 = [sum(v) for v in zip(mean_uncertainties_2, uncertainties)]

    # retraining test 3: low uncertain data first
    sorted_idxs = reversed(sorted_idxs)
    for cur_idx, sorted_idx in enumerate(sorted_idxs):
        subject_dataset.inps[cur_idx] = temp_inps[sorted_idx]
        subject_dataset.outs[cur_idx] = temp_outs[sorted_idx]
    subject_data_loader = data.DataLoader(subject_dataset, batch_size=partition_size, shuffle=False, drop_last=True)
    accuracies, uncertainties = retrain(phar_model_3, subject_data_loader)
    mean_accuracies_3 = [sum(v) for v in zip(mean_accuracies_3, accuracies)]
    mean_uncertainties_3 = [sum(v) for v in zip(mean_uncertainties_3, uncertainties)]

    pbar.update(1)
pbar.close()

print('\n=== Random Split ===\n')
for idx in range(n_results):
    print(f'Accuracy\t{mean_accuracies_1[idx] / len(subject_names):0.4f}\t'
          f'Uncertainty\t{mean_uncertainties_1[idx] / len(subject_names):0.5f}')

print('\n\n=== High Uncertain Data First ===\n')
for idx in range(n_results):
    print(f'Accuracy\t{mean_accuracies_2[idx] / len(subject_names):0.4f}\t'
          f'Uncertainty\t{mean_uncertainties_2[idx] / len(subject_names):0.5f}')

print('\n\n=== Low Uncertain Data First ===\n')
for idx in range(n_results):
    print(f'Accuracy\t{mean_accuracies_3[idx] / len(subject_names):0.4f}\t'
          f'Uncertainty\t{mean_uncertainties_3[idx] / len(subject_names):0.5f}')
