"""
Human Activity Recognition for Robot House Multiview Dataset 
@author Mohammad Abadi <m.bamorovvat@gmail.com>
"""

import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchmetrics import Accuracy
from dataset_preparation import VideoDataset
from models import C3D_model, R2Plus1D_model, R3D_model

import neptune.new as neptune

run = neptune.init(
    project="m.bamorovat/Abbas-RHHAR",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMzk0NzVhZi05MmJmLTQzZWEtYjU2Ni01YjUwMzA4MmZkN2MifQ==",
)

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#  device = torch.device("cpu")
print("Device being used:", device)

nEpochs = 250  # Number of epochs for training   100
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 20  # Run on test set every nTestInterval epochs
snapshot = 50  # Store a model every snapshot epochs
lr = 1e-3  # Learning rate
BatchSize = 30  # Bach Size
num_workers = 32  # Number of worker
num_classes = 14
# global val_acc, best_acc1
val_acc = None
# best_acc1 = 0

dataset = 'rhhar' # Options: hmdb51 or ucf101 or rhhar
view = 'RobotView' # Options: 'OmniView' , 'FrontView' , 'BackView' , 'RobotView'

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'C3D'  # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset + '-' + view


def train_model(dataset=dataset, view = view , save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
            :param test_interval:
            :param useTest:
            :param save_epoch:
            :param num_epochs:
            :param lr:
            :param num_classes:
            :param save_dir:
            :param dataset:
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
        pretrained = 'Yes'
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
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)  # optim.Adam
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, view=view, split='train', clip_len=16),
                                  batch_size=BatchSize, shuffle=True, num_workers=num_workers)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, view=view, split='val',  clip_len=16),
                                  batch_size=BatchSize, num_workers=num_workers)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, view=view, split='test', clip_len=16),
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

    best_acc1 = 0

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                #post_act = torch.nn.Softmax(dim=1)
                #preds1 = post_act(outputs)
                top5_pred_classes = probs.topk(k=5).indices[0]
                top1_pred_classes = probs.topk(k=1).indices[0]

                # Map the predicted classes to the label names
                top5_pred_class_names = [labels[int(i)] for i in top5_pred_classes]
                top1_pred_class_names = [labels[int(i)] for i in top1_pred_classes]
                # print("Top 5 predicted labels: %s" % ", ".join(top5_pred_class_names))

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]


            if phase == 'train':
                writer.add_scalar('train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('train_acc_epoch', epoch_acc, epoch)
                #writer.add_scalar('train_top_5', top5_pred_class_names, epoch)
                #writer.add_scalar('train_top_1', top1_pred_class_names, epoch)
                run["train_loss"].log(epoch_loss, epoch)
                run["train_acc"].log(epoch_acc, epoch)
                run["train_top_5"].log(top5_pred_class_names, epoch)
                run["train_top_1"].log(top1_pred_class_names, epoch)

            else:
                writer.add_scalar('val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('val_acc_epoch', epoch_acc, epoch)
                run["val_loss"].log(epoch_loss, epoch)
                run["val_acc"].log(epoch_acc, epoch)
                run["val_top_5"].log(top5_pred_class_names, epoch)
                run["val_top_1"].log(top1_pred_class_names, epoch)

                val_acc = epoch_acc

            print("[{}] Epoch: {}/{} Loss: {} Acc: {} Top5: {} Top1: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc, top5_pred_class_names, top1_pred_class_names))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        # evaluate on validation set
        acc1 = val_acc

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # if epoch % save_epoch == (save_epoch - 1):
        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'best_acc1': best_acc1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '.pth.tar')))
            print("Saved Epoch Number is: ", epoch)

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                #post_act = torch.nn.Softmax(dim=1)
                #preds1 = post_act(outputs)
                top5_pred_classes = probs.topk(k=5).indices[0]
                top1_pred_classes = probs.topk(k=1).indices[0]

                # Map the predicted classes to the label names
                top5_pred_class_names = [labels[int(i)] for i in top5_pred_classes]
                top1_pred_class_names = [labels[int(i)] for i in top1_pred_classes]
                # print("Top 5 predicted labels: %s" % ", ".join(top5_pred_class_names))

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('test_acc_epoch', epoch_acc, epoch)
            run["test_loss"].log(epoch_loss, epoch)
            run["test_acc"].log(epoch_acc, epoch)
            run["test_top_5"].log(top5_pred_class_names, epoch)
            run["test_top_1"].log(top1_pred_class_names, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {} Top5: {} Top1: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc, top5_pred_class_names, top1_pred_class_names))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()
    run.stop()


if __name__ == "__main__":
    train_model()

