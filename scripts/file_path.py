import os


class Path(object):
    @staticmethod
    def db_dir(database, view):
        har_path = '/beegfs/general/mb19aag'
        if database == 'rhhar':
            # folder that contains class labels
            root_dir = os.path.join(har_path, 'RHHAR', view)

            # Save preprocess data into output_dir
            output_dir = os.path.join(har_path, 'RHHAR' + '_' + view, 'rhhar' + '_' + view + '_' + 'var')
            # print('Output Directory: ', output_dir)

            split_file_dir = os.path.join(har_path, 'RHHAR' + '_' + view)
            # print(split_file_dir)

            return root_dir, output_dir, split_file_dir

        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir(model):
        if model == 'C3D':
            return '/home/mb19aag/test/PytorchVideoRecognition/saved_models/c3d-pretrained.pth'
        elif model == 'R3D':
            return '/home/mb19aag/test/RH_HAR/pretrained_models/r3d_18-b3b3357e.pth'
        elif model == 'R2Plus1D':
            return '/home/mb19aag/test/RH_HAR/pretrained_models/r2plus1d_18-91a641e6.pth'
        elif model == 'Slow_Fast':
            return '/home/mb19aag/test/RH_HAR/pretrained_models/SLOWFAST_4x16_R50.pkl'
        elif model == 'X3D':
            return '/home/mb19aag/test/RH_HAR/pretrained_models/'
