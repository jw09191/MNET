from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.datamodules.components.aistpp_dataset import AISTPPDataset


class AISTPPDataModule(LightningDataModule):
    def __init__(self,
                 data_path,
                 music_length,
                 seed_m_length,
                 predict_length,
                 train_test_split,
                 batch_size=8,
                 num_workers=4,
                 pin_memory=True,
                 **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        dataset = AISTPPDataset(data_path, music_length, seed_m_length, predict_length)
        self.data_train, self.data_test = random_split(dataset, train_test_split)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False
        )


if __name__ == '__main__':
    datamodule = AISTPPDataModule(
        data_path='/hdd_sda1/jw09191/data/AIST++',
        sample_length=120,
        train_test_split=[1368, 40],
        batch_size=32,
        num_workers=16,
        pin_memory=True
    )