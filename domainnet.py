import torch
import os
import torchvision
from torchvision import transforms
from torchvision.transforms.transforms import RandomResizedCrop, Resize, ToPILImage 

domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((80, 80)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

transform_val = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class DomainNet(torch.utils.data.Dataset):
    def __init__(self, client_id, train):
        super().__init__()
        assert(client_id < len(domains))

        self.client_id = client_id
        self.domain = domains[self.client_id]
        self.transform = transform
        
        if train:
            lines = open(f"../DomainNet/{self.domain}_train.txt", "r").readlines()
            self.transform = transform
        else:
            lines = open(f"../DomainNet/{self.domain}_test.txt", "r").readlines()
            self.transform = transform_val

        self.examples = {}
        self.labels = set()
        for i, l in enumerate(lines):
            img_path, label = l.split()
            img_path = os.path.join("../DomainNet", img_path)
            label = int(label)

            self.examples[i] = (img_path, label)
            self.labels.add(label)

    def __getitem__(self, idx):
        p, y = self.examples[idx]
        x = torchvision.io.read_image(p)
        if not self.transform is None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.examples)

def get_domainnet(batch_size, num_workers, num_clients):
    assert(num_clients <= 6)
    tr_dl = []
    train_sizes = []
    num_classes = []
    for c in range(num_clients):
        ds = DomainNet(c, True)
        dl = torch.utils.data.DataLoader(
            ds, 
            batch_size=batch_size, 
            pin_memory=True, 
            num_workers=num_workers
        )
        tr_dl.append(dl)
        train_sizes.append(len(ds))
        num_classes.append(len(ds.labels))

    ts_dl = []
    for c in range(num_clients):
        ds = DomainNet(c, False)
        dl = torch.utils.data.DataLoader(
            ds, 
            batch_size=batch_size, 
            pin_memory=True, 
            num_workers=num_workers
        )
        ts_dl.append(dl)

    return tr_dl, ts_dl, train_sizes

if __name__ == "__main__":
    ds = DomainNet(0, True)
    x, y = ds[0]
    print(x.shape, y)

    ds = DomainNet(0, False)
    x, y = ds[0]
    print(x.shape, y)

    tr_dl, ts_dl, train_sizes = get_domainnet(6, 32, 2)
    print(train_sizes)