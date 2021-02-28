import torch
from torch.utils.data import Dataset


class RedditDataset(Dataset):
    def __init__(self, labels, posts):
        super().__init__()
        self.labels = labels
        self.posts = posts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        labels = torch.tensor(self.labels[item], dtype=torch.long)
        posts = torch.tensor(self.posts[item], dtype=torch.float32)
        return [labels, posts]