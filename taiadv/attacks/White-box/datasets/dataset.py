from .utils import transform_options, dataset_options
from torch.utils.data import DataLoader
from torchvision import transforms


class DatasetGenerator():
    def __init__(self, eval_bs=256, seed=0, n_workers=4,
                 test_d_type='CIFAR10', test_path='../../datasets/',
                 **kwargs):

        if test_d_type not in transform_options:
            raise('Unknown Dataset')

        self.eval_bs = eval_bs
        self.n_workers = n_workers
        self.test_path = test_path

        test_tf = transform_options[test_d_type]["test_transform"]
        test_tf = transforms.Compose(test_tf)
        self.test_set = dataset_options[test_d_type](test_path, test_tf,
                                                     True, kwargs)
        self.test_set_length = len(self.test_set)

    def get_loader(self, train_shuffle=True):
        test_loader = DataLoader(dataset=self.test_set, pin_memory=True,
                                 batch_size=self.eval_bs, drop_last=False,
                                 num_workers=self.n_workers, shuffle=False)
        return test_loader
