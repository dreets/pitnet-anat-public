# global imports
import albumentations as album
import numpy as np
import os
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from ast import literal_eval
from efficientnet_pytorch import EfficientNet
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models

# local imports
from utils_loss import Loss
from utils_main import Paths
from utils_main import PdUtils
from utils_main import Terms
from utils_main import device_initialise

# strongly typed
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor
from typing import Dict
from typing import List
from typing import Tuple


def main():
    pass


def get_dt_image_landmarks(
        str_image: str,
        tp_structures: Tuple = Terms.Structures.tp_structures,
        csv_coordinates: Path = Paths.csv_coordinates,
) -> Dict[str, Tuple[float, float]]:
    """read the landmarks for a given {str_image}"""
    dt_image_landmarks: Dict[str, Tuple[float, float]] = {}
    df_coordinates: DataFrame = pd.read_csv(csv_coordinates)
    df_images: DataFrame = PdUtils(df=df_coordinates).subset(column="str_image", value=str_image)
    for str_structure in tp_structures:
        df_structure: DataFrame = PdUtils(df=df_images).subset(column="str_structure", value=str_structure)
        ls_centroid: str = PdUtils(df=df_structure).ls_values("str_centroid")
        if ls_centroid:
            str_centroid: str = ls_centroid[0]
        else:
            str_centroid: str = "(0.0, 0.0)"
        dt_image_landmarks[str_structure] = literal_eval(str_centroid)
    return dt_image_landmarks


def dt_loaders_initialise(
        int_fold: int,
        int_height: int,
        int_width: int,
        int_batch: int,
        tp_structures_mrk: Tuple[str] = Terms.Structures.tp_structures_mrk,
        tp_structures_msk: Tuple[str] = Terms.Structures.tp_structures_msk,
        path_images: Path = Paths.path_input_images,
        path_masks: Path = Paths.path_input_masks,
        csv_split: Path = Paths.csv_split,
        csv_coordinates: Path = Paths.csv_coordinates,
        bool_train_shuffle: bool = True,
        bool_transforms: bool = True,
) -> Dict[str, DataLoader]:
    """initialise data loader as a (train, val) dictionary"""
    dt_transformers: Dict[str, Compose] = dt_transformers_initialise(int_width=int_width, int_height=int_height)
    dt_loader: Dict[str, DataLoader] = {}
    for str_split in Terms.tp_split:
        dataset: Dataset = LandmarksDataset(
            int_fold=int_fold,
            int_height=int_height,
            int_width=int_width,
            str_split=str_split,
            dt_transforms=dt_transformers,
            tp_structures_mrk=tp_structures_mrk,
            tp_structures_msk=tp_structures_msk,
            path_images=path_images,
            path_masks=path_masks,
            csv_split=csv_split,
            csv_coordinates=csv_coordinates,
            bool_transforms=bool_transforms,
        )
        bool_train_shuffle = bool_train_shuffle and (str_split == Terms.trn)
        dt_loader[str_split]: DataLoader = DataLoader(dataset, batch_size=int_batch, shuffle=bool_train_shuffle)
    return dt_loader


def dt_transformers_initialise(
    int_height: int,
    int_width: int,
) -> Dict[str, Compose]:
    """initialise the image augementations"""
    dt_transformers: Dict[str, Compose] = {
        Terms.trn: album.Compose(
            [
                album.ShiftScaleRotate(
                    shift_limit=(-0.1, 0.1),
                    scale_limit=(0, 0.2),
                    rotate_limit=(-11, 11),
                    always_apply=True,
                    p=1,
                ),
                album.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.05,
                    hue=0.05,
                    always_apply=True,
                    p=1
                ),
                album.Resize(height=int_height, width=int_width),
                ToTensorV2(),
            ],
            keypoint_params=album.KeypointParams(format="xy", remove_invisible=False)),
        Terms.val: album.Compose(
            [
                album.Resize(height=int_height, width=int_width),
                ToTensorV2(),
            ],
            keypoint_params=album.KeypointParams(format="xy", remove_invisible=False),
        )
    }
    return dt_transformers


class LandmarksDataset(Dataset):
    """create Dataset type for segmentation task"""
    def __init__(
            self,
            int_fold: int,
            int_width: int,
            int_height: int,
            dt_transforms: Dict[str, Compose],
            str_split: str = Terms.trn,
            tp_structures_mrk: Tuple[str] = Terms.Structures.tp_structures_mrk,
            tp_structures_msk: Tuple[str] = Terms.Structures.tp_structures_msk,
            path_images: Path = Paths.path_input_images,
            path_masks: Path = Paths.path_input_masks,
            csv_split: Path = Paths.csv_split,
            csv_coordinates: Path = Paths.csv_coordinates,
            bool_transforms: bool = True,
    ):
        self.int_fold: int = int_fold
        self.int_height: int = int_height
        self.int_width: int = int_width
        self.str_split: str = str_split
        self.tp_structures_mrk: Tuple[str] = tp_structures_mrk
        self.tp_structures_msk: Tuple[str] = tp_structures_msk
        self.tp_structures: Tuple[str] = tp_structures_mrk + tp_structures_msk
        self.path_images: Path = path_images
        self.path_masks: Path = path_masks
        self.csv_split: Path = csv_split
        self.csv_coordinates: Path = csv_coordinates
        self.ls_images: List[str] = self.find_dt_image_split()[str_split]
        self.dt_transforms: Dict[str, Compose] = dt_transforms
        self.bool_transforms: bool = bool_transforms

    def __len__(self) -> int:
        return len(self.ls_images)

    def __getitem__(self, int_index) -> Tuple[Tensor, Tensor, Tensor]:
        """return the tensor form of images and the corresponding mask"""
        str_image: str = self.ls_images[int_index]
        dt_mrk: Dict[str, Tuple[float, float]] = get_dt_image_landmarks(
            str_image=str_image,
            tp_structures=self.tp_structures,
            csv_coordinates=self.csv_coordinates,
        )
        ay_img: ndarray = self.ay_create_img(str_image=str_image)
        ay_mrk: ndarray = self.ay_create_mrk(dt_mrk=dt_mrk, tp_shape=ay_img.shape)
        ay_msk: ndarray = self.ay_create_msk(str_image=str_image)
        ts_img, ls_mrk, ts_msk = self.ay_augmentations(ay_img=ay_img, ay_mrk=ay_mrk, ay_msk=ay_msk)
        ts_mrk = self.ts_create_mrk(ls_mrk=ls_mrk, dt_mrk=dt_mrk)
        return ts_img, ts_mrk, ts_msk

    def ay_create_img(self, str_image: str) -> ndarray:
        """create the img array for a given {int_index}"""
        path_img: str = os.path.join(self.path_images, str_image)
        ay_img: ndarray = np.array(Image.open(path_img).convert("RGB"), dtype=np.float32) / 255
        return ay_img

    @staticmethod
    def ay_create_mrk(dt_mrk: Dict[str, Tuple[float, float]], tp_shape: Tuple[float, float]) -> ndarray:
        """create the landmark array for a given {int_index}"""
        ls_mrk: List[Tuple] = []
        for str_structure in dt_mrk.keys():
            x_centroid: float = dt_mrk[str_structure][0] * tp_shape[1]
            y_centroid: float = dt_mrk[str_structure][1] * tp_shape[0]
            ls_mrk.append((x_centroid, y_centroid))
        ay_mrk: ndarray = np.array(ls_mrk).astype("float32")
        return ay_mrk

    def ay_create_msk(self, str_image: str) -> ndarray:
        """create the img array for a given {int_index}"""
        ls_ay_msks: List[ndarray] = []
        for str_structure in self.tp_structures:
            path_msk: str = os.path.join(self.path_masks, str_structure, str_image)
            ay_msk: ndarray = np.array(Image.open(path_msk).convert("L"), dtype=np.float32) / 255
            ls_ay_msks.append(ay_msk)
        ls_ay_msks: List[ndarray] = self.ay_overlap_find_all(ls_ay_msks=ls_ay_msks)
        ay_msks: ndarray = np.stack(ls_ay_msks, axis=-1)
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

    def ay_augmentations(self, ay_img: ndarray, ay_mrk: ndarray, ay_msk: ndarray) -> Tuple[Tensor, List, Tensor]:
        """apply {self.dt_transformations} on {ay_img} & {ay_mrk)"""
        if self.bool_transforms:
            augmentations = self.dt_transforms[self.str_split](image=ay_img, mask=ay_msk, keypoints=ay_mrk)
            ts_img: Tensor = augmentations["image"]
            ts_msk: Tensor = augmentations["mask"]
            ls_mrk: List = augmentations["keypoints"]
        else:
            augmentations = self.dt_transforms[Terms.val](image=ay_img, mask=ay_msk, keypoints=ay_mrk)
            ts_img: Tensor = augmentations["image"]
            ts_msk: Tensor = augmentations["mask"]
            ls_mrk: List = augmentations["keypoints"]
        return ts_img, ls_mrk, ts_msk

    def ts_create_mrk(self, ls_mrk: List, dt_mrk: Dict[str, Tuple[float, float]]) -> Tensor:
        ls_mrk_new: List[Tuple[float, float]] = []
        for int_index, str_structure in enumerate(self.tp_structures):
            if dt_mrk[str_structure] != (0.0, 0.0):
                ls_mrk_new.append(ls_mrk[int_index])
            else:
                ls_mrk_new.append((np.nan, np.nan))

        ts_mrk: Tensor = Tensor(ls_mrk_new)
        return ts_mrk

    def find_dt_image_split(self) -> Dict[str, List[str]]:
        """find the list of images corresponding to training and validation"""
        df_split: DataFrame = pd.read_csv(self.csv_split)
        df_coordinates: DataFrame = pd.read_csv(self.csv_coordinates)
        ls_videos_trn: List[int] = df_split[df_split.fold != self.int_fold].video.to_list()
        ls_videos_val: List[int] = df_split[df_split.fold == self.int_fold].video.to_list()
        ls_images_trn: List[str] = list(df_coordinates[df_coordinates.video.isin(ls_videos_trn)].image.unique())
        ls_images_val: List[str] = list(df_coordinates[df_coordinates.video.isin(ls_videos_val)].image.unique())
        dt_split: Dict[str, List[str]] = {Terms.trn: ls_images_trn, Terms.val: ls_images_val}
        return dt_split

    def ts_mrks_msks_prds_clean(
            self,
            ts_prds_mrks: Tensor,
            ts_prds_msks: Tensor,
            ts_mrks: Tensor,
            ts_msks: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        int_batch: int = ts_mrks.shape[0]
        int_structures: int = len(self.tp_structures)
        for int_i in range(int_batch):
            for int_j in range(int_structures * 2):
                if not (int_j % 2):
                    int_structure: int = int(int_j / 2)
                    if (self.tp_structures[int_structure] in self.tp_structures_msk) or (
                            np.isnan(ts_mrks[int_i][int_j]).item() and np.isnan(ts_mrks[int_i][int_j + 1]).item()):
                        ts_prds_mrks[int_i][int_j] = 0
                        ts_prds_mrks[int_i][int_j + 1] = 0
                        ts_mrks[int_i][int_j] = 0
                        ts_mrks[int_i][int_j + 1] = 0
                    ts_mrks[int_i][int_j] /= self.int_width
                    ts_mrks[int_i][int_j + 1] /= self.int_height
            for int_structure, str_structure in enumerate(self.tp_structures):
                if str_structure in self.tp_structures_mrk:
                    ay_msk_zero: ndarray = np.zeros_like(ts_prds_msks[int_i][int_structure].detach())
                    ts_prds_msks[int_i][int_structure] = Tensor(ay_msk_zero)
                    ts_msks[int_i].permute(2, 0, 1)[int_structure] = Tensor(ay_msk_zero)
        return ts_prds_mrks, ts_prds_msks, ts_mrks, ts_msks

    def bool_prediction_in_mask(
            self,
            x_prd: float,
            y_prd: float,
            ts_msk: Tensor,
    ) -> bool:
        """check if the predicted centroid is within the ground-truth mask"""
        int_x: int = int(np.round(x_prd * self.int_width))
        int_y: int = int(np.round(y_prd * self.int_height))

        if int_x < 0 or int_x >= self.int_width or int_y < 0 or int_y >= self.int_height:
            return False
        else:
            if ts_msk[int_y][int_x]:  # coordinates are reversed
                return True
            else:
                return False


def find_tensor_markers(
        imgs: Tensor,
        mrks: Tensor,
        msks: Tensor,
        model,
        loader: DataLoader,
        device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    ts_imgs: Tensor = imgs.float().to(device)
    ts_mrks: Tensor = mrks.view(mrks.size(0), -1).float().to(device=device)
    ts_msks: Tensor = msks.float().to(device)
    ts_prds_msks, ts_prds_mrks = model(ts_imgs)
    ts_prds_mrks, ts_prds_msks, ts_mrks, ts_msks = loader.dataset.ts_mrks_msks_prds_clean(
        ts_prds_mrks=ts_prds_mrks.cpu(),
        ts_prds_msks=ts_prds_msks.cpu(),
        ts_mrks=ts_mrks.cpu(),
        ts_msks=ts_msks.cpu(),
    )
    return ts_imgs, ts_mrks, ts_msks, ts_prds_mrks, ts_prds_msks


def model_initialise(str_model: str, int_classes: int, device=device_initialise()):
    """initialise model {str_model}"""
    model = None
    aux_params = dict(
        pooling="avg",  # one of 'avg', 'max'
        dropout=0.3,  # dropout ratio, default is None
        activation="sigmoid",  # activation function, default is None
        classes=int_classes * 2,  # define number of output labels
    )

    if str_model == Terms.Models.Encoders.resnet18:
        model = ResNet18Marks(int_classes * 2)
    if str_model == Terms.Models.Encoders.resnet50:
        model = ResNet50Marks(int_classes * 2)
    if str_model == Terms.Models.Encoders.efficientb3:
        model = EfficientNetB3Marks(int_classes * 2)
    if str_model == Terms.Models.unetv2:
        model = smp.UnetPlusPlus(
            encoder_name=Terms.Models.Encoders.efficientb3,
            classes=int_classes,
            aux_params=aux_params,
        )
    if str_model == Terms.Models.unet:
        model = smp.Unet(
            encoder_name=Terms.Models.Encoders.efficientb3,
            classes=int_classes,
            aux_params=aux_params,
        )
    return model.to(device)


def optimiser_initialise(str_optimiser: str, model_parameters, flt_rate: float):
    """initialise optimiser {str_optimiser}"""
    optimiser = None
    if str_optimiser == Terms.Models.Optimisers.adam:
        optimiser = torch.optim.Adam(model_parameters, lr=flt_rate)
    return optimiser


def loss_initialise(str_loss: str):
    loss = None
    if str_loss == Terms.Models.Losses.mse:
        loss = Loss(str_mode="marks", str_loss="mse", str_type="multi")
    return loss


class ResNet18Marks(torch.nn.Module):
    def __init__(self, int_classes=20):
        super().__init__()
        self.model_name = "resnet18marks"
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7,),
            stride=(2,),
            padding=3,
            bias=False
        )
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=int_classes)

    def forward(self, x):
        x = self.model(x)
        return torch.tensor([]), x


class ResNet50Marks(torch.nn.Module):
    def __init__(self, int_classes=20):
        super().__init__()
        self.model_name = "resnet50marks"
        self.model = models.resnet50(pretrained=True)
        self.model.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7,),
            stride=(2,),
            padding=3,
            bias=False
        )
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=int_classes)

    def forward(self, x):
        x = self.model(x)
        return torch.tensor([]), x


class EfficientNetB3Marks(torch.nn.Module):
    def __init__(self, int_classes=20):
        super().__init__()
        self.model_name = "efficientnetb3"
        self.model = EfficientNet.from_pretrained("efficientnet-b3")
        self.model.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7,),
            stride=(2,),
            padding=3,
            bias=False
        )
        self.model._fc = torch.nn.Linear(in_features=self.model._fc.in_features, out_features=int_classes)

    def forward(self, x):
        x = self.model(x)
        return torch.tensor([]), x


if __name__ == "__main__":
    main()
