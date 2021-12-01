import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
from functools import partial
from typing import Optional, Callable
from torch.utils.model_zoo import tqdm
import PIL
import tarfile

import os
import os.path
import logging
import torchvision.datasets.utils as utils
from torchvision import transforms
import random
from sklearn.model_selection import train_test_split
from domainnet import get_domainnet

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class CustomTensorDataset(data.TensorDataset):
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) + (index,)


class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        # if self.train:
        #     data = mnist_dataobj.train_data
        #     target = mnist_dataobj.train_labels
        # else:
        #     data = mnist_dataobj.test_data
        #     target = mnist_dataobj.test_labels

        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        # print("mnist img:", img)
        # print("mnist target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class FashionMNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        # if self.train:
        #     data = mnist_dataobj.train_data
        #     target = mnist_dataobj.train_labels
        # else:
        #     data = mnist_dataobj.test_data
        #     target = mnist_dataobj.test_labels

        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        # print("mnist img:", img)
        # print("mnist target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class SVHN_custom(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train is True:
            # svhn_dataobj1 = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            # svhn_dataobj2 = SVHN(self.root, 'extra', self.transform, self.target_transform, self.download)
            # data = np.concatenate((svhn_dataobj1.data, svhn_dataobj2.data), axis=0)
            # target = np.concatenate((svhn_dataobj1.labels, svhn_dataobj2.labels), axis=0)

            svhn_dataobj = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            data = svhn_dataobj.data
            target = svhn_dataobj.labels
        else:
            svhn_dataobj = SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
            data = svhn_dataobj.data
            target = svhn_dataobj.labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        # print("svhn data:", data)
        # print("len svhn data:", len(data))
        # print("type svhn data:", type(data))
        # print("svhn target:", target)
        # print("type svhn target", type(target))
        return data, target

    # def truncate_channel(self, index):
    #     for i in range(index.shape[0]):
    #         gs_index = index[i]
    #         self.data[gs_index, :, :, 1] = 0.0
    #         self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # print("svhn img:", img)
        # print("svhn target:", target)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


# torchvision CelebA
class CelebA_custom(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``. If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(self, root, dataidxs=None, split="train", target_type="attr", transform=None,
                 target_transform=None, download=False):
        import pandas
        super(CelebA_custom, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[split.lower()]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split is None else (splits[1] == split)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        self.gender_index = self.attr_names.index('Male')
        self.dataidxs = dataidxs
        if self.dataidxs is None:
            self.target = self.attr[:, self.gender_index:self.gender_index + 1].reshape(-1)
        else:
            self.target = self.attr[self.dataidxs, self.gender_index:self.gender_index + 1].reshape(-1)

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        if self.dataidxs is None:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

            target = []
            for t in self.target_type:
                if t == "attr":
                    target.append(self.attr[index, self.gender_index])
                elif t == "identity":
                    target.append(self.identity[index, 0])
                elif t == "bbox":
                    target.append(self.bbox[index, :])
                elif t == "landmarks":
                    target.append(self.landmarks_align[index, :])
                else:
                    # TODO: refactor with utils.verify_str_arg
                    raise ValueError("Target type \"{}\" is not recognized.".format(t))
        else:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[self.dataidxs[index]]))

            target = []
            for t in self.target_type:
                if t == "attr":
                    target.append(self.attr[self.dataidxs[index], self.gender_index])
                elif t == "identity":
                    target.append(self.identity[self.dataidxs[index], 0])
                elif t == "bbox":
                    target.append(self.bbox[self.dataidxs[index], :])
                elif t == "landmarks":
                    target.append(self.landmarks_align[self.dataidxs[index], :])
                else:
                    # TODO: refactor with utils.verify_str_arg
                    raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)
        #print("target[0]:", target[0])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
        #print("celeba target:", target)
        return X, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.attr)
        else:
            return len(self.dataidxs)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)



class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None) -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:   # download the file
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")

def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")


def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")


def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")


def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")


def _is_gzip(filename: str) -> bool:
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename: str) -> bool:
    return filename.endswith(".zip")


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)

class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        self.dataidxs = dataidxs

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.targets = self.targets[self.dataidxs]        


    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        mkdirs(self.raw_folder)
        mkdirs(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)

    def __len__(self):
        return len(self.data)


class Generated(MNIST):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        self.dataidxs = dataidxs

        if self.train:
            self.data = np.load("data/generated/X_train.npy")
            self.targets = np.load("data/generated/y_train.npy")
        else:
            self.data = np.load("data/generated/X_test.npy")
            self.targets = np.load("data/generated/y_test.npy")            

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.targets = self.targets[self.dataidxs]        


    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self):
        return len(self.data)



class genData(MNIST):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __getitem__(self,index):
        data, target = self.data[index], self.targets[index]
        return data, target
    def __len__(self):
        return len(self.data)

def split_dataset_disjoint_labels(num_clients, dataset, num_groups=2):
    assert(num_clients % num_groups == 0)
    
    labels = [y for x, y in dataset]
    labels_to_idx = {}
    for i, l in enumerate(labels):
        if l not in labels_to_idx:
            labels_to_idx[l] = []
        labels_to_idx[l].append(i)

    num_classes = len(set(labels))
    assert(num_classes % num_groups == 0)

    labels_to_chunks = {}
    chunk_counter = {}
    for k, v in labels_to_idx.items():
        labels_to_chunks[k] = np.split(np.array(v), num_clients // num_groups)
        chunk_counter[k] = 0

    group_ids = []
    for gid in range(num_groups):
        group_ids += [gid] * (num_clients // num_groups)
    random.shuffle(group_ids)

    unique_labels = list(range(num_classes))
    group_id_to_labels = {}
    random.shuffle(unique_labels)
    for i, l in enumerate(unique_labels):
        if i % num_groups not in group_id_to_labels:
            group_id_to_labels[i % num_groups] = []
        group_id_to_labels[i % num_groups].append(l)

    client_id_to_idx = {}
    for c in range(num_clients):
        for l in unique_labels:
            if l in group_id_to_labels[group_ids[c]]:
                if client_id_to_idx.get(c, False) == False:
                    client_id_to_idx[c] = [] 
                client_id_to_idx[c].extend(list(labels_to_chunks[l][chunk_counter[l]])) 
                chunk_counter[l] += 1
    return client_id_to_idx

def classwise_subset(total_dataset, num_clients, num_groups, test_split=0.1):
    client_id_to_idx = split_dataset_disjoint_labels(num_clients, total_dataset, num_groups)
    train, test = {}, {}
    train_sizes = torch.zeros((num_clients,))

    for c, indices in client_id_to_idx.items():
        train_idx, test_idx = train_test_split(indices, test_size=test_split)
        train[c] = train_idx
        test[c] = test_idx 

        train_sizes[c] = len(train_idx)
    return train, test, train_sizes

def SplitCIFAR10(num_clients, batch_size):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = CIFAR10(
        root="./data", train=False, download=True, transform=transform_val
    )
    # Make Overlapping non-IID client splits
    total = torch.utils.data.ConcatDataset([trainset, testset])
    train_idx, test_idx, train_sizes = classwise_subset(
        total,
        num_clients,
        5,
        0.1
    )
    cifar_test_loader_list = []
    cifar_train_loader_list = []
    for c in range(num_clients):
        # ts = torch.utils.data.Subset(total, train_idx[c])
        train_dl = torch.utils.data.DataLoader(
            total, 
            batch_size=batch_size, 
            pin_memory=False, 
            sampler=torch.utils.data.SubsetRandomSampler(train_idx[c])
        )
        cifar_train_loader_list.append(train_dl)

        test_dl = torch.utils.data.DataLoader(
            total, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=False,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx[c])
        )
        cifar_test_loader_list.append(test_dl)
    return cifar_train_loader_list, cifar_test_loader_list, train_sizes

non_iid_50_home = "/home/surya/Documents/Projects/distributed-ml-research/split_nn_ucb/non_iid_50/scripts/tasks/"

from functools import partial
from operator import sub
import os
import pdb
import glob
import numpy as np
from misc.utils import *
from dataclasses import dataclass
import torch
import math


@dataclass
class Arguments:
    base_dir: str
    num_clients: int


class DataLoader:
    """ Data Loader

    Loading data for the corresponding clients

    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)

    Modified by: 
        Surya Kant Sahu (surya.oju@pm.me)
    """

    def __init__(self, args):
        self.args = args
        self.base_dir = args.base_dir
        self.did_to_dname = {
            0: 'cifar10',
            1: 'cifar100',
            2: 'mnist',
            3: 'svhn',
            4: 'fashion_mnist',
            5: 'traffic_sign',
            6: 'face_scrub',
            7: 'not_mnist',
        }

        files = os.listdir(self.args.base_dir)
        files = ["_".join(f.replace(".npy", "").split("_")[:-1])
                 for f in files]
        files = ['cifar100_0', 'cifar10_0', 'fashion_mnist_0', 'mnist_0', 'not_mnist_0', 'svhn_0', 'face_scrub_0', 'traffic_sign_0', 'cifar100_1', 'cifar10_1', 'fashion_mnist_1', 'mnist_1', 'not_mnist_1', 'svhn_1', 'face_scrub_1', 'traffic_sign_1']
        assert(self.args.num_clients <= len(files))
        self.task_set = {
            k: [v] for k, v in enumerate(files[:self.args.num_clients])
        }

    def get_train(self, cid, task_id):
        return load_task(self.base_dir, self.task_set[cid][task_id]+'_train.npy').item()

    def get_valid(self, cid, task_id):
        valid = load_task(
            self.base_dir, self.task_set[cid][task_id]+'_valid.npy').item()
        return valid['x_valid'], valid['y_valid']

    def get_test(self, cid, task_id):
        x_test_list = []
        y_test_list = []
        for tid, task in enumerate(self.task_set[cid]):
            if tid <= task_id:
                test = load_task(self.base_dir, task+'_test.npy').item()
                x_test_list.append(test['x_test'])
                y_test_list.append(test['y_test'])
        return x_test_list, y_test_list


class ShuffledCycle:
    def __init__(self, indices):
        self.indices = indices
        self.i = 0
        random_shuffle(77, self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.indices):
            self.i = 0
            random_shuffle(77, self.indices)
        self.i += 1
        return self.indices[self.i-1]


class NonIID50Train(torch.utils.data.Dataset):
    def __init__(self, num_clients, client_id, version):
        assert(version in ['v1', 'v2'])

        self.num_clients = num_clients
        self.client_id = client_id

        args = Arguments(os.path.join(non_iid_50_home, f"non_iid_50_{version}/non_iid_50/"), num_clients)
        self.dl = dl = DataLoader(args)
        self.data = {}
        for k in range(num_clients):
            d = dl.get_train(k, 0)
            self.data[k] = {
                "x": d['x_train'],
                "y": d['y_train']
            }
            
        self.ds_sizes = {k: v["x"].shape[0] for k, v in self.data.items()}
        self.index_cycles = {k: ShuffledCycle(
            list(range(v))) for k, v in self.ds_sizes.items()}

    def __len__(self):
        return self.ds_sizes[self.client_id]

    def __getitem__(self, *args):
        c = self.client_id
        idx = next(self.index_cycles[c])
        xc = self.data[c]["x"][idx].transpose(2, 0, 1)
        yc = one_hot_to_int(self.data[c]["y"][idx])

        return xc, yc

    def shuffle(self):
        pass


class NonIID50Test(torch.utils.data.Dataset):
    def __init__(self, num_clients, client_id, version):
        assert(version in ['v1', 'v2'])

        self.num_clients = num_clients
        self.client_id = client_id

        args = Arguments(os.path.join(non_iid_50_home, f"non_iid_50_{version}/non_iid_50/"), num_clients)
        self.dl = dl = DataLoader(args)
        self.data = {}
        for k in range(num_clients):
            x, y = dl.get_test(k, 0)
            self.data[k] = {
                "x": x[0],
                "y": y[0]
            }

        self.ds_sizes = {k: v["x"].shape[0] for k, v in self.data.items()}
        self.index_cycles = {k: ShuffledCycle(
            list(range(v))) for k, v in self.ds_sizes.items()}

    def __len__(self):
        return self.ds_sizes[self.client_id]

    def __getitem__(self, *args):
        c = self.client_id
        idx = next(self.index_cycles[c])
        xc = self.data[c]["x"][idx].transpose(2, 0, 1)
        yc = one_hot_to_int(self.data[c]["y"][idx])

        return xc, yc

    def shuffle(self):
        pass


def one_hot_to_int(onehot):
    for i in range(len(onehot)):
        if onehot[i] > 0.:
            return i

def non_iid_50_collate_fn(batches):
    xb, yb = [], []
    for x, y in batches:
        xb.append(x)
        yb.append(y)
    xb = torch.FloatTensor(np.stack(xb)).contiguous()
    yb = torch.LongTensor(np.stack(yb)).contiguous()
    return xb, yb

def get_non_iid_50_v1(batch_size, num_workers, num_clients):
    tr_dl = []
    for i in range(num_clients):
        tr_ds = NonIID50Train(num_clients, i, version="v1")
        dataset_sizes = [int(i) for i in tr_ds.ds_sizes]
        dl = torch.utils.data.DataLoader(
            tr_ds, batch_size, num_workers=num_workers, collate_fn=non_iid_50_collate_fn, pin_memory=True)
        tr_dl.append(dl)

    ts_dl = []
    for i in range(num_clients):
        ts_ds = NonIID50Test(num_clients, client_id=i, version="v1")
        dl = torch.utils.data.DataLoader(
            ts_ds, batch_size, num_workers=num_workers, collate_fn=non_iid_50_collate_fn, pin_memory=True)
        ts_dl.append(dl)
    return tr_dl, ts_dl, dataset_sizes

def get_non_iid_50_v2(batch_size, num_workers, num_clients):
    tr_dl = []
    for i in range(num_clients):
        tr_ds = NonIID50Train(num_clients, i, batch_size, version="v2")
        dataset_sizes = [int(i) for i in tr_ds.ds_sizes]
        dl = torch.utils.data.DataLoader(
            tr_ds, batch_size, num_workers=num_workers, collate_fn=non_iid_50_collate_fn, pin_memory=True)
        tr_dl.append(dl)

    ts_dl = []
    for i in range(num_clients):
        ts_ds = NonIID50Test(num_clients, client_id=i, version="v2")
        dl = torch.utils.data.DataLoader(
            ts_ds, batch_size, num_workers=num_workers, collate_fn=non_iid_50_collate_fn, pin_memory=True)
        ts_dl.append(dl)
    return tr_dl, ts_dl, dataset_sizes
