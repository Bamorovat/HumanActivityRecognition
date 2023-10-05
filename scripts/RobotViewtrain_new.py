import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchmetrics import Accuracy
from dataset_preparation import VideoDataset
from models import C3D_model, R2Plus1D_model, R3D_model, Slow_Fast_model, X3D_model
from torchnet.engine import Engine
import torchnet as tnt
import pandas as pd
import neptune.new as neptune
import numpy as np

run = neptune.init(
    project="m.bamorovat/Top",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMzk0NzVhZi05MmJmLTQzZWEtYjU2Ni01YjUwMzA4MmZkN2MifQ==",
)

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device being used:", device)

nEpochs = 250  # Number of epochs for training   100
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 15  # Run on test set every nTestInterval epochs
snapshot = 50  # Store a model every snapshot epochs
lr = 2e-3  # Learning rate
BatchSize = 30  # Bach Size
num_workers = 32  # Number of worker
num_classes = 14  # Number of Classes
dataset = 'rhhar'  # Options: hmdb51 or ucf101 or rhhar
view = 'RobotView'  # Options: 'OmniView' , 'FrontView' , 'BackView' , 'RobotView'
val_acc = None

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'Slow_Fast'  # Options: C3D or R2Plus1D or R3D or Slow_Fast or X3D
saveName = modelName + '-' + dataset + '-' + view + '-' + 'Top'


def processor(sample):
    data, labels, training = sample
    data, labels = data.to(device), labels.to(device)
    model.train(training)
    classes = model(data)
    loss = loss_criterion(classes, labels)

    return loss, classes


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    meter_confusion.reset()


def on_forward(state):
    meter_accuracy.add(state['output'].detach().cpu(), state['sample'][1])
    # meter_confusion.add(state['output'].detach().cpu(), state['sample'][1])
    meter_loss.add(state['loss'].item())


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    results['train_loss'].append(meter_loss.value()[0])
    results['train_top1_accuracy'].append(meter_accuracy.value()[0])
    results['train_top5_accuracy'].append(meter_accuracy.value()[1])
   
    writer.add_scalar('train_loss_epoch', meter_loss.value()[0], state['epoch'])
    writer.add_scalar('train_top1_accuracy', meter_accuracy.value()[0], state['epoch'])
    writer.add_scalar('train_top5_accuracy', meter_accuracy.value()[1], state['epoch'])
    run["train_loss"].log(meter_loss.value()[0], state['epoch'])
    run["train_top1_accuracy"].log(meter_accuracy.value()[0], state['epoch'])
    run["train_top5_accuracy"].log(meter_accuracy.value()[1], state['epoch'])
    # run["train_confusion"].log(meter_confusion.value())

    print('[Epoch %d] Training Loss: %.4f Top1 Accuracy: %.2f%% Top5 Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0], meter_accuracy.value()[1]))

    reset_meters()

    # val
    with torch.no_grad():
        engine.test(processor, val_dataloader)

    results['val_loss'].append(meter_loss.value()[0])
    results['val_top1_accuracy'].append(meter_accuracy.value()[0])
    results['val_top5_accuracy'].append(meter_accuracy.value()[1])

    writer.add_scalar('Validation_loss_epoch', meter_loss.value()[0], state['epoch'])
    writer.add_scalar('val_top1_accuracy', meter_accuracy.value()[0], state['epoch'])
    writer.add_scalar('val_top5_accuracy', meter_accuracy.value()[1], state['epoch'])
    run["Validation_loss"].log(meter_loss.value()[0], state['epoch'])
    run["val_top1_accuracy"].log(meter_accuracy.value()[0], state['epoch'])
    run["val_top5_accuracy"].log(meter_accuracy.value()[1], state['epoch'])
    # run["val_confusion"].log(meter_confusion.value())

    print('[Epoch %d] Valing Loss: %.4f Top1 Accuracy: %.2f%% Top5 Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0], meter_accuracy.value()[1]))

    # save best model
    global best_accuracy
    if meter_accuracy.value()[0] > best_accuracy:
        torch.save(model.state_dict(), os.path.join(save_dir, 'models', saveName + '_101' + '.pth.tar'))
        best_accuracy = meter_accuracy.value()[0]

    scheduler.step(meter_loss.value()[0])
    reset_meters()

    # test
    with torch.no_grad():
        engine.test(processor, test_dataloader)

    results['test_loss'].append(meter_loss.value()[0])
    results['test_top1_accuracy'].append(meter_accuracy.value()[0])
    results['test_top5_accuracy'].append(meter_accuracy.value()[1])

    writer.add_scalar('test_loss_epoch', meter_loss.value()[0], state['epoch'])
    writer.add_scalar('test_top1_accuracy', meter_accuracy.value()[0], state['epoch'])
    writer.add_scalar('test_top5_accuracy', meter_accuracy.value()[1], state['epoch'])
    run["test_loss"].log(meter_loss.value()[0], state['epoch'])
    run["test_top1_accuracy"].log(meter_accuracy.value()[0], state['epoch'])
    run["test_top5_accuracy"].log(meter_accuracy.value()[1], state['epoch'])
    # run["test_confusion"].log(meter_confusion.value())

    print('[Epoch %d] Testing Loss: %.4f Top1 Accuracy: %.2f%% Top5 Accuracy: %.2f%%' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0], meter_accuracy.value()[1],))

    # save statistics at each epoch
    data_frame = pd.DataFrame(
        data={'train_loss': results['train_loss'], 'train_top1_accuracy': results['train_top1_accuracy'],
              'train_top5_accuracy': results['train_top5_accuracy'], 'val_loss': results['val_loss'], 'val_top1_accuracy': results['val_top1_accuracy'],
              'val_top5_accuracy': results['val_top5_accuracy'], 'test_loss': results['test_loss'], 'test_top1_accuracy': results['test_top1_accuracy'],
              'test_top5_accuracy': results['test_top5_accuracy']},
        index=range(1, state['epoch'] + 1))
    data_frame.to_csv(os.path.join(save_dir, 'models', saveName + 'results.csv'), index_label='epoch')


if __name__ == "__main__":

    results = {'train_loss': [], 'train_top1_accuracy': [], 'train_top5_accuracy': [], 'val_loss': [],
               'val_top1_accuracy': [], 'val_top5_accuracy': [], 'test_loss': [], 'test_top1_accuracy': [],
               'test_top5_accuracy': []}

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        pretrained = 'No'
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        pretrained = 'No'
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        pretrained = 'No'
        train_params = model.parameters()
    elif modelName == 'Slow_Fast':
        model = Slow_Fast_model.resnet50(class_num=num_classes)
        pretrained = 'No'
        train_params = model.parameters()
    elif modelName == 'X3D':
        X3D_VERSION = 'S'
        pretrained = 'No'
        model = X3D_model.generate_model(x3d_version=X3D_VERSION, n_classes=num_classes, n_input_channels=3,
                                         dropout=0.5, base_bn_splits=1)
        model.replace_logits(num_classes)
        train_params = model.parameters()
    else:
        print('The Model is not Defined !!!')
        raise NotImplementedError

    if modelName == 'X3D':
        optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)
        loss_criterion = nn.BCEWithLogitsLoss()
    elif modelName != 'X3D':
        # criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
        loss_criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
        optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    print("Training {} from scratch...".format(modelName))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model = model.to(device)
    # criterion.to(device)
    model = nn.DataParallel(model)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, view=view, split='train', clip_len=16),
                                  batch_size=BatchSize, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, view=view, split='val', clip_len=16),
                                batch_size=BatchSize, num_workers=num_workers)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, view=view, split='test', clip_len=16),
                                 batch_size=BatchSize, num_workers=num_workers)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    params = {'Learning_rate': lr,
              'Optimiser': optimizer,
              'NEpochs': nEpochs,
              'num_workers': num_workers,
              'Batch_size': BatchSize,
              'Test_Batch_size': 1,
              'Training_data': len(train_dataloader.dataset),
              'Validation_data': len(val_dataloader.dataset),
              'Test_data': len(test_dataloader.dataset),
              'Frame_Length': 16,
              'Frame_Height': 128,
              'Frame_Width': 171,
              'Frame_Crop': 112,
              'Model': modelName,
              'Dataset': dataset,
              'View': view,
              'Pretrained': pretrained,
              'job_name': '{}_{}_{}'.format(dataset, modelName, optimizer)}
    run["parameters"] = params

    best_accuracy = 0

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    meter_confusion = tnt.meter.ConfusionMeter(num_classes, normalized=True)

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_dataloader, maxepoch=nEpochs, optimizer=optimizer)


