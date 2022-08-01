from torch.utils.data import DataLoader, Dataset, random_split
from torchvideo import transforms
from pandas import read_csv
from PIL import Image
import os
from torchvision.transforms import InterpolationMode


class VideoClassificationDataset(Dataset):
    def __init__(self, frame_dir, csv_path, resize):
        super(VideoClassificationDataset, self).__init__()
        self.frame_dir = frame_dir
        self.csv_path = csv_path
        self.resize = resize
        self.df = read_csv(csv_path,encoding='utf-8')
        self.transformer = transforms.Compose([
            transforms.RandomHorizontalFlipVideo(),
            transforms.ResizeVideo(self.resize,interpolation=InterpolationMode.BICUBIC),
            transforms.PILVideoToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        frame_dir = self.df['framepath'][idx]
        label = self.df['label'][idx]
        frame_list = []
        for frame_name in os.listdir(frame_dir):
            frame_path = os.path.join(frame_dir, frame_name)
            pil_img = Image.open(frame_path)
            frame_list.append(pil_img)
        video = self.transformer(frame_list)
        return video, label


def get_dataloader(frame_dir, csv_path, resize, batch_size, num_workers, train_percent):
    dataset = VideoClassificationDataset(frame_dir, csv_path, resize)
    num_sample = len(dataset)
    num_train = int(train_percent * num_sample)
    num_valid = num_sample - num_train
    train_ds, valid_ds = random_split(dataset, [num_train, num_valid])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_dl, valid_dl, len(dataset), len(train_ds), len(valid_ds)

