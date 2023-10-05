"""
Human Activity Recognition for Robot House Multiview Dataset 
@author Mohammad Abadi <m.bamorovvat@gmail.com>
"""

import os
from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
# from mypath import Path
# from pytorch-video-recognition.mypath import Path

import sys
sys.path.append('/home/abbas/Deep_Learning/pytorch-video-recognition')
from file_path import Path


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
            view='RobotView', 'FrontView', 'BackView', 'OmniView'
    """

    def __init__(self, dataset='rhhar', view='OmniView', split='train', clip_len=16, preprocess=True):
        self.root_dir, self.output_dir, self.split_file_dir = Path.db_dir(dataset, view)
        folder = os.path.join(self.output_dir, split)
        self.view = view
        # folder = os.path.join(self.root_dir, self.view)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        # if (not self.check_preprocess()) or preprocess:
        if not self.check_preprocess():
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []

        for label in sorted(os.listdir(folder)):
           for fname in os.listdir(os.path.join(folder, label)):
               self.fnames.append(os.path.join(folder, label, fname))
               labels.append(label)

        # assert len(labels) == len(self.fnames)  # 1st difference
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if os.path.exists(self.root_dir):
            # print('integrity is OK')
            return True
        else:
            # print('integrity is Not OK')
            return False

    def check_preprocess(self):
        if os.path.exists(os.path.join(self.output_dir + '_' + 'average', 'train')) and os.path.exists(os.path.join(self.output_dir + '_' + 'average', 'test')) and os.path.exists(os.path.join(self.output_dir + '_' + 'average', 'val')):
            # print('Check PreProcessing is OK')
            return True
        else:
            # print('Check PreProcessing is NOt OK')
            return False

    def preprocess(self):
        if not os.path.exists(self.output_dir + '_' + 'average'):
            os.mkdir(self.output_dir + '_' + 'average')


        motion_situation = 'average'  # average , sub , mean

        if self.split == 'train':
            train_video_files = []
            if not os.path.exists(os.path.join(self.output_dir + '_' + motion_situation, 'train')):
                os.mkdir(os.path.join(self.output_dir + '_' + motion_situation, 'train'))
            for line in open(os.path.join(self.split_file_dir, self.view + '_' + 'trainlist.txt'), 'r'):
                train_video_files.append(line.split()[0])
        elif self.split == 'val':
            val_video_files = []
            if not os.path.exists(os.path.join(self.output_dir + '_' + motion_situation, 'val')):
                os.mkdir(os.path.join(self.output_dir + '_' + motion_situation, 'val'))
            for line in open(os.path.join(self.split_file_dir, self.view + '_' + 'vallist.txt'), 'r'):
                val_video_files.append(line.split()[0])
        elif self.split == 'test':
            test_video_files = []
            if not os.path.exists(os.path.join(self.output_dir + '_' + motion_situation, 'test')):
                os.mkdir(os.path.join(self.output_dir + '_' + motion_situation, 'test'))
            for line in open(os.path.join(self.split_file_dir, self.view + '_' + 'testlist.txt'), 'r'):
                test_video_files.append(line.split()[0])


        if self.split == 'train':
            for video in train_video_files:
                file = format(video.split('/')[0])
                videoo = format(video.split('/')[1])
                if not os.path.exists(os.path.join(self.output_dir + '_' + motion_situation, 'train', file)):
                    os.mkdir(os.path.join(self.output_dir + '_' + motion_situation, 'train', file))
                #self.process_video(videoo, file, os.path.join(self.output_dir, 'train', file))
                self.process_video_multiframes(videoo, file, os.path.join(self.output_dir + '_' + motion_situation, 'train', file))
        elif self.split == 'val':
            for video in val_video_files:
                file = format(video.split('/')[0])
                videoo = format(video.split('/')[1])
                if not os.path.exists(os.path.join(self.output_dir + '_' + motion_situation, 'val', file)):
                    os.mkdir(os.path.join(self.output_dir + '_' + motion_situation, 'val', file))
                #self.process_video(videoo, file, os.path.join(self.output_dir, 'val', file))
                self.process_video_multiframes(videoo, file, os.path.join(self.output_dir + '_' + motion_situation, 'val', file))
        elif self.split == 'test':
            for video in test_video_files:
                file = format(video.split('/')[0])
                videoo = format(video.split('/')[1])
                if not os.path.exists(os.path.join(self.output_dir + '_' + motion_situation, 'test', file)):
                    os.mkdir(os.path.join(self.output_dir + '_' + motion_situation, 'test', file))
                #self.process_video(videoo, file, os.path.join(self.output_dir, 'test', file))
                self.process_video_multiframes(videoo, file, os.path.join(self.output_dir + '_' + motion_situation, 'test', file))

        print('Preprocessing finished.')

    def process_video_multiframes(self, video, action_name, save_dir):
        print('Preprocess {}'.format(video))
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        # print(video_filename)
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        print('Captured file name: ', os.path.join(self.root_dir, action_name, video))
        if not os.path.exists (os.path.join(self.root_dir, action_name)):
            print('The video file is not existed')
        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Video frame count: ', frame_count)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 5
        Frame_Rate = 16
        if frame_count // EXTRACT_FREQUENCY <= Frame_Rate:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= Frame_Rate:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= Frame_Rate:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):

            for a in range(EXTRACT_FREQUENCY):
                retaining, frame = capture.read()
                if frame is None:
                    continue

                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))

                if a == 0:
                        avg_image = frame
                        #mean_image = frame
                        #sub_image = frame
                else:
                        alpha = 1.0 / (a + 1)
                        beta = 1.0 - alpha
                        avg_image = cv2.addWeighted(frame, alpha, avg_image, beta, 0.0)
                        #mean_image = ((frame * 0.3) + (mean_image * 1.8)) * 0.45  # / 2
                        #sub_image = (frame - (sub_image * 0.97))  # * 1.01

                #retaining, frame = capture.read()
                #if frame is None:
                #    continue

            cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=avg_image)
            #cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=mean_image)
            #cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=sub_image)

            i += 1
            count = count + EXTRACT_FREQUENCY

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def process_video(self, video, action_name, save_dir):
        print('Preprocess {}'.format(video))
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        # print(video_filename)
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        print('Captured file name: ', os.path.join(self.root_dir, action_name, video))
        if not os.path.exists (os.path.join(self.root_dir, action_name)):
            print('The video file is not existed')
        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Video frame count: ', frame_count)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        while frame_count <= 16:
            frame_count += 1
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        i = 0
        frame = None

        for _, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
            i += 1

        while i < 16:
            buffer[i] = frame
            i += 1

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='rhhar', split='test', clip_len=16, preprocess=True)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
