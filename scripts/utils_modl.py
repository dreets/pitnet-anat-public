# global imports
import albumentations as album
import os
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from ast import literal_eval
from pathlib import Path
from PIL import Image
from torch import load
from torch import optim

# local imports
from utils_loss import Loss
from utils_main import Paths
from utils_main import Terms
from utils_main import device_initialise
from utils_main import str2none

# strongly typed
from albumentations import Compose
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Dict
from typing import List
from typing import Tuple


def main():
    pass


def loss_initialise(str_loss: str, str_mode: str, str_type: str = "mask"):
    return Loss(str_loss=str_loss, str_mode=str_mode, str_type=str_type)


def model_initialise(
        str_model: str = Terms.Models.unet,
        str_encoder: str = Terms.Models.Encoders.efficientb3,
        int_channels: int = 3,
        int_classes: int = 1,
        str_activation: str = None,
        device=device_initialise(),
):
    """initialise {str_model_name} with {str_encoder}"""
    model = None
    if str_model == Terms.Models.unet:
        model = smp.Unet(
            encoder_name=str_encoder,
            in_channels=int_channels,
            classes=int_classes,
            activation=str2none(str_activation),
        )
    elif str_model == Terms.Models.deeplabv3:
        model = smp.DeepLabV3(
            encoder_name=str_encoder,
            in_channels=int_channels,
            classes=int_classes,
            activation=str2none(str_activation),
        )
    elif str_model == Terms.Models.deeplabv4:
        model = smp.DeepLabV3Plus(
            encoder_name=str_encoder,
            in_channels=int_channels,
            classes=int_classes,
            activation=str2none(str_activation),
        )
    elif str_model == Terms.Models.linknet:
        model = smp.Linknet(
            encoder_name=str_encoder,
            in_channels=int_channels,
            classes=int_classes,
            activation=str2none(str_activation),
        )
    elif str_model == Terms.Models.pan:
        model = smp.PAN(
            encoder_name=str_encoder,
            in_channels=int_channels,
            classes=int_classes,
            activation=str2none(str_activation),
        )
    elif str_model == Terms.Models.unetv2:
        model = smp.UnetPlusPlus(
            encoder_name=str_encoder,
            in_channels=int_channels,
            classes=int_classes,
            activation=str2none(str_activation),
        )
    elif str_model == Terms.Models.pspnet:
        model = smp.PSPNet(
            encoder_name=str_encoder,
            in_channels=int_channels,
            classes=int_classes,
            activation=str2none(str_activation),
        )
    elif str_model == Terms.Models.manet:
        model = smp.MAnet(
            encoder_name=str_encoder,
            in_channels=int_channels,
            classes=int_classes,
            activation=str2none(str_activation),
        )
    elif str_model == Terms.Models.fpn:
        model = smp.FPN(
            encoder_name=str_encoder,
            in_channels=int_channels,
            classes=int_classes,
            activation=str2none(str_activation),
        )
    model.to(device)
    return model


def optimiser_initialise(str_optimiser: str, model_parameters, flt_rate: float):
    """initialise optimiser {str_optimiser}"""
    optimiser = None
    if str_optimiser == Terms.Models.Optimisers.adam:
        optimiser = optim.Adam(model_parameters, lr=flt_rate)
    return optimiser


def dt_loaders_initialise(
        int_fold: int = 0,
        str_mode: str = Terms.Models.Losses.str_multiclass,
        int_height: int = 736,  # resized from 720 for compatibility
        int_width: int = 1280,
        tp_structures: Tuple[str] = (Terms.Structures.sella,),
        int_batch: int = 10,
        path_images: Path = Paths.path_input_images,
        path_masks: Path = Paths.path_input_masks,
        csv_split: Path = Paths.csv_split,
) -> Dict[str, DataLoader]:
    """initialise data loader as a (train, val) dictionary"""
    dt_transformers: Dict[str, Compose] = dt_transformers_initialise(int_width=int_width, int_height=int_height)
    dt_loader: Dict[str, DataLoader] = {}
    for str_split in Terms.tp_split_test:
        dataset: Dataset = SegmentationDataset(
            int_fold=int_fold,
            str_mode=str_mode,
            str_split=str_split,
            tp_structures=tp_structures,
            dt_transforms=dt_transformers,
            path_images=path_images,
            path_masks=path_masks,
            csv_split=csv_split,
        )
        dt_loader[str_split]: DataLoader = DataLoader(dataset, batch_size=int_batch, shuffle=(str_split == Terms.trn))
    return dt_loader


def dt_transformers_initialise(
    int_height: int = 736,  # resized from 720 for compatibility
    int_width: int = 1280,
) -> Dict[str, Compose]:
    """augmentation to increase data variability"""
    dt_transformers: Dict[str, Compose] = {
        Terms.trn: album.Compose([
            album.ShiftScaleRotate(
                shift_limit=(-0.1, 0.1),
                scale_limit=(-0.1, 0.2),
                rotate_limit=(-11, 11),
                always_apply=True,
                p=1,
            ),
            album.Resize(height=int_height, width=int_width),
            album.ColorJitter(always_apply=True, p=1),
            ToTensorV2(),
        ],),
        Terms.val: album.Compose([
            album.Resize(height=int_height, width=int_width),
            ToTensorV2(),
        ],),
        Terms.tst: album.Compose([
            album.Resize(height=int_height, width=int_width),
            ToTensorV2(),
        ],)
    }
    return dt_transformers


class SegmentationDataset(Dataset):
    """create dataset type for segmentation task"""
    def __init__(
            self,
            int_fold: int = 0,
            str_split: str = Terms.trn,
            str_mode: str = Terms.Models.Losses.str_multiclass,
            tp_structures: Tuple[str] = None,
            dt_transforms: Dict[str, Compose] = None,
            path_images: Path = Paths.path_input_images,
            path_masks: Path = Paths.path_input_masks,
            csv_split: Path = Paths.csv_split,
            csv_coordinates: Path = Paths.csv_coordinates,
    ):
        self.int_fold: int = int_fold
        self.str_split: str = str_split
        self.tp_structures: Tuple[str] = tp_structures
        self.path_images: Path = path_images
        self.path_masks: Path = path_masks
        self.dt_transforms: Dict[str, Compose] = dt_transforms
        self.csv_split: Path = csv_split
        self.csv_coordinates: Path = csv_coordinates
        self.ls_images: List[str] = self.find_dt_image_split()[str_split]
        self.str_mode: str = str_mode

    def __len__(self) -> int:
        return len(self.ls_images)

    def __getitem__(self, int_index) -> Tuple[Tensor, Tensor]:
        """return the tensor form of images and the corresponding mask"""
        ay_img: ndarray = self.ay_create_img(int_index)
        ay_msk: ndarray = self.ay_create_msk(int_index)
        ts_img, ts_msk = self.ts_augmentations(ay_img, ay_msk)
        return ts_img, ts_msk

    def find_dt_image_split(self) -> Dict[str, List[str]]:
        """find the list of images corresponding to training and validation"""
        df_split: DataFrame = pd.read_csv(self.csv_split)
        df_coordinates: DataFrame = pd.read_csv(self.csv_coordinates)
        ls_videos_trn: List[int] = df_split[df_split.fold != self.int_fold].video.to_list()
        ls_videos_tst: List[int] = df_split[df_split.fold == -1].video.to_list()
        ls_videos_trn = list(set(ls_videos_trn).symmetric_difference(set(ls_videos_tst)))
        ls_videos_val: List[int] = df_split[df_split.fold == self.int_fold].video.to_list()
        ls_images_trn: List[str] = list(df_coordinates[df_coordinates.video.isin(ls_videos_trn)].image.unique())
        ls_images_val: List[str] = list(df_coordinates[df_coordinates.video.isin(ls_videos_val)].image.unique())
        ls_images_tst: List[str] = list(df_coordinates[df_coordinates.video.isin(ls_videos_tst)].image.unique())
        dt_split: Dict[str, List[str]] = {Terms.trn: ls_images_trn, Terms.val: ls_images_val, Terms.test: ls_images_tst}
        return dt_split

    def ts_augmentations(self, ay_img, ay_msk) -> Tuple[Tensor, Tensor]:
        """apply {self.dt_transformations} on {ay_img} & {ay_msk), and transform them into tensors"""
        if self.dt_transforms:
            augmentations = self.dt_transforms[self.str_split](image=ay_img, mask=ay_msk)
            ts_img: Tensor = augmentations[Terms.img]
            ts_msk: Tensor = augmentations[Terms.msk]
        else:
            ts_img: Tensor = Tensor(ay_img)
            ts_msk: Tensor = Tensor(ay_msk)

        if self.str_mode == Terms.Models.Losses.str_multilabel:
            ts_msk = ts_msk.permute(2, 0, 1)
        return ts_img, ts_msk

    def ay_create_img(self, int_index) -> ndarray:
        """create the img array for a given {int_index}"""
        path_img: str = os.path.join(self.path_images, self.ls_images[int_index])
        ay_img: ndarray = np.array(Image.open(path_img).convert("RGB"), dtype=np.float32) / 255
        return ay_img

    def ay_create_msk(self, int_index: int) -> ndarray:
        """create the mask array for a given {int_index}"""
        ls_ay_msks: List[ndarray] = []
        for str_structure in self.tp_structures:
            path_msk: str = os.path.join(self.path_masks, str_structure, self.ls_images[int_index])
            ay_msk: ndarray = np.array(Image.open(path_msk).convert("L"), dtype=np.float32) / 255
            ls_ay_msks.append(ay_msk)
            ls_ay_msks: List[ndarray] = self.ay_overlap_find_all(ls_ay_msks=ls_ay_msks)

        if self.str_mode == Terms.Models.Losses.str_binary:
            ay_msks: ndarray = ls_ay_msks[0]
        elif self.str_mode == Terms.Models.Losses.str_multiclass:
            ay_msks: ndarray = self.ay_create_multiclass(ls_ay_msks=ls_ay_msks)
        else:
            ay_msks: ndarray = np.stack(ls_ay_msks, axis=-1)
        return ay_msks

    @staticmethod
    def ay_create_multiclass(ls_ay_msks: List[ndarray]) -> ndarray:
        """create an array such that all values values in the image are unique (0 is the background class)"""
        ay_msks: ndarray = np.zeros([ls_ay_msks[0].shape[0], ls_ay_msks[0].shape[1]])
        for int_class, ay_msk in enumerate(ls_ay_msks):
            ay_msks += ls_ay_msks[int_class] * (int_class + 1)
        ay_msks -= 1  # -1 is the background class but don't want to calculate loss on
        return ay_msks

    def ay_overlap_find_all(self, ls_ay_msks: List[ndarray]) -> List[ndarray]:
        """find if there is any overlap between any two masks in {ls_ay_msks}"""
        for int_i in range(len(ls_ay_msks)):
            for int_j in range(len(ls_ay_msks)):
                if int_i > int_j:
                    ls_ay_msks[int_i], ls_ay_msks[int_j] = self.ay_overlap_find(
                        ay_a=ls_ay_msks[int_i],
                        ay_b=ls_ay_msks[int_j],
                    )
        return ls_ay_msks

    @staticmethod
    def ay_overlap_find(ay_a: ndarray, ay_b: ndarray) -> Tuple[ndarray, ndarray]:
        """find if there is any overlap between two masks {ay_A} and {ay_B}, and remove {ay_B} overlaps"""
        ay_overlap: ndarray = np.column_stack(np.where(np.logical_and(ay_a == 1, ay_b == 1)))
        for int_x, int_y in ay_overlap:
            ay_a[int_x, int_y] = 1
            ay_b[int_x, int_y] = 0
        return ay_a, ay_b


def dataloader_load(flt_model_number: float, csv_model_map: str = Paths.csv_model_map_msk) -> Dict[str, DataLoader]:
    """load data using parameters of {flt_model_number}"""
    df_model_map: DataFrame = pd.read_csv(csv_model_map)
    df_model: DataFrame = df_model_map[df_model_map[Terms.Headings.flt_model_number] == flt_model_number]
    dt_loaders: Dict[str, DataLoader] = dt_loaders_initialise(
        int_fold=df_model["int_fold"].to_list()[0],
        str_mode=df_model["str_mode"].to_list()[0],
        int_height=df_model["int_height"].to_list()[0],
        int_width=df_model["int_width"].to_list()[0],
        tp_structures=literal_eval(df_model["tp_str_structures"].to_list()[0]),
        int_batch=1,
        path_images=Paths.path_input_images,
        path_masks=Paths.path_input_masks,
        csv_split=Paths.csv_split,
    )
    return dt_loaders


def model_load(
        flt_model_number: float,
        path_models: str = Paths.path_models,
        csv_model_map: str = Paths.csv_model_map_msk
):
    """load model {flt_model_number}"""
    df_model_map: DataFrame = pd.read_csv(csv_model_map)
    df_model: DataFrame = df_model_map[df_model_map[Terms.Headings.flt_model_number] == flt_model_number]
    model = model_initialise(
        str_model=df_model["str_model"].to_list()[0],
        str_encoder=df_model["str_encoder"].to_list()[0],
        int_channels=df_model["int_channels"].to_list()[0],
        int_classes=df_model["int_classes"].to_list()[0],
        str_activation=df_model["str_activation"].to_list()[0],
        device=device_initialise(),
    )
    if int(df_model["int_classes"].to_list()[0]) == 1:
        str_model_path = os.path.join(path_models, f"{int(flt_model_number)}.pth")
    else:
        str_model_path = os.path.join(path_models, f"{flt_model_number}.pth")
    model.load_state_dict(load(str_model_path))
    return model


if __name__ == "__main__":
    main()
