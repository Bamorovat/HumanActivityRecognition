import torch
import torch.nn as nn
# from mypath import Path
from file_path import Path


class C3DMultiview(nn.Module):
    """
    The C3D_multiview network.
    """

    def __init__(self, num_classes, pretrained=False):
        self.pretrained = pretrained
        super(C3DMultiview, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.batch1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.batch2 = nn.BatchNorm3d(128)

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.batch3a = nn.BatchNorm3d(256)
        self.batch3b = nn.BatchNorm3d(256)

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.batch4a = nn.BatchNorm3d(512)
        self.batch4b = nn.BatchNorm3d(512)

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.batch5a = nn.BatchNorm3d(512)
        self.batch5b = nn.BatchNorm3d(512)

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.batch6 = nn.BatchNorm1d(4096)
        self.batch7 = nn.BatchNorm1d(4096)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        # self.batch = nn.BatchNorm1d(4096)
        self.__init_weight()

        if self.pretrained:
            self.__load_pretrained_weights()

    def forward(self, x, y):

        x = self.relu(self.conv1(x))
        x = self.batch1(x)
        x = self.pool1(x)

        y = self.relu(self.conv1(y))
        y = self.batch1(y)
        y = self.pool1(y)

        x, y = self.__lateral(x, y)

        x = self.relu(self.conv2(x))
        x = self.batch2(x)
        x = self.pool2(x)

        y = self.relu(self.conv2(y))
        y = self.batch2(y)
        y = self.pool2(y)
        
        x, y = self.__lateral(x, y)

        x = self.relu(self.conv3a(x))
        x = self.batch3a(x)
        
        y = self.relu(self.conv3a(y))
        y = self.batch3a(y)
        
        x, y = self.__lateral(x, y)
        
        x = self.relu(self.conv3b(x))
        x = self.batch3b(x)
        x = self.pool3(x)
       
        y = self.relu(self.conv3b(y))
        y = self.batch3b(y)
        y = self.pool3(y)
        
        x, y = self.__lateral(x, y)

        x = self.relu(self.conv4a(x))
        x = self.batch4a(x)
        
        y = self.relu(self.conv4a(y))
        y = self.batch4a(y)
        
        x, y = self.__lateral(x, y)
        
        x = self.relu(self.conv4b(x))
        x = self.batch4b(x)
        x = self.pool4(x)        
        
        y = self.relu(self.conv4b(y))
        y = self.batch4b(y)
        y = self.pool4(y)
        
        x, y = self.__lateral(x, y)

        x = self.relu(self.conv5a(x))
        x = self.batch5a(x)
        
        y = self.relu(self.conv5a(y))
        y = self.batch5a(y)
        
        x, y = self.__lateral(x, y)
        
        x = self.relu(self.conv5b(x))
        x = self.batch5b(x)
        x = self.pool5(x)
       
        y = self.relu(self.conv5b(y))
        y = self.batch5b(y)
        y = self.pool5(y)
        
        # x, y = self.__lateral(x, y)

        x = x.view(-1, 8192)
        y = y.view(-1, 8192)
        
        x_ave = x * 0.5
        y_ave = y * 0.5
        ave = x_ave.add(y_ave)
        
        x_y_ave = torch.cat([x, y, ave], dim=1)

        x_y_ave = self.relu(self.fc6(x_y_ave))
        x_y_ave = self.batch6(x_y_ave)
        x_y_ave = self.dropout(x_y_ave)
        x_y_ave = self.relu(self.fc7(x_y_ave))
        x_y_ave = self.batch7(x_y_ave)
        x_y_ave = self.dropout(x_y_ave)

        logits = self.fc8(x_y_ave)

        return logits

    def __lateral(self, x, y):
        x_ave = x * 0.5
        y_ave = y * 0.5
        ave = x_ave.add(y_ave)
        x = torch.cat([x, ave], dim=1)
        y = torch.cat([y, ave], dim=1)
        return x, y

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir('C3D'))
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

