# global imports
import argparse
import git
import os
import re
import shutil
import torch

# strongly type
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from pandas import DataFrame


def main():
    pass


class Paths:
    """pre-defined paths for ease of use"""
    # base directories
    path_directory: Path = os.path.dirname(os.path.realpath(__file__))
    path_base: Path = git.Repo(path_directory, search_parent_directories=True).git.rev_parse("--show-toplevel")
    path_inputs: Path = path_base.joinpath("inputs")
    path_outputs: Path = path_base.joinpath("outputs")
    path_scripts: Path = path_base.joinpath("scripts")

    # input child directories
    path_input_images: Path = path_inputs.joinpath("images")
    path_input_masks: Path = path_inputs.joinpath("masks")
    path_input_texts: Path = os.path.join(path_inputs, "texts")

    # input files
    csv_coordinates: Path = path_input_texts.joinpath("coordinates.csv")
    csv_split: Path = path_input_texts.joinpath("split.csv")

    # output child directories
    path_output_texts: Path = path_outputs.joinpath("texts")
    path_models: Path = path_outputs.joinpath("models")
    path_models_mrk: Path = path_outputs.joinpath("marks")
    path_models_msk: Path = path_outputs.joinpath("masks")
    path_models_mul: Path = path_outputs.joinpath("multi")

    # output history directories
    path_history: Path = path_output_texts.joinpath("history")
    path_history_mrk: Path = path_history.joinpath("marks")
    path_history_msk: Path = path_history.joinpath("masks")
    path_history_mul: Path = path_history.joinpath("multi")

    # output result files
    path_results: Path = path_output_texts.joinpath("results")
    csv_mean: Path = path_results.joinpath("mean.csv")
    csv_model_map_msk: Path = path_results.joinpath("model_map_masks.csv")
    csv_model_map_mrk: Path = path_results.joinpath("model_map_marks.csv")
    csv_model_map_mul: Path = path_results.joinpath("model_map_multi.csv")


class Terms:
    """pre-defined terms for ease of use"""
    image: str = "image"
    img: str = "image"
    mask: str = "mask"
    msk: str = "mask"
    mark: str = "mark"
    mrk: str = "mark"

    fold: str = "fold"
    tp_folds: Tuple[int] = (0, 1, 2, 3, 4)

    train: str = "train"
    trn: str = "train"
    val: str = "val"
    test: str = "test"
    tst: str = "test"
    tp_split: Tuple[str] = (trn, val)
    tp_split_test: Tuple[str] = (trn, val, tst)

    class Headings:
        coordinates: str = "coordinates"
        frame: str = "frame"
        int_epoch: str = "int_epoch"
        int_fold: str = "int_fold"
        flt_model_number: str = "flt_model_number"
        video: str = "video"
        tp_str_structures: str = "tp_str_structures"
        tp_important: Tuple[str] = (
            "str_structure",
            tp_str_structures,
            "str_model",
            "str_encoder",
            "str_loss",
            "str_mode",
            "flt_rate",
            "str_activation",
            "int_batch",
            "int_epochs",
            "str_optimiser"
        )

    class Metrics:
        accuracy: str = "accuracy"
        acc: str = "accuracy"
        iou: str = "iou"
        dce: str = "dice"
        dice: str = "dice"
        recall: str = "recall"
        rec: str = "recall"
        precision: str = "precision"
        pre: str = "precision"
        tp_metrics: Tuple[str] = (acc, rec, pre, dce, iou)
        tp_thresholds: Tuple[float] = (0.50, 0.40, 0.30, 0.20, 0.10, 0.05)

    class Models:
        deeplabv4: str = "deeplabv3+"
        deeplabv3: str = "deeplabv3"
        fpn: str = "fpn"
        linknet: str = "linknet"
        pan: str = "pan"
        pspnet: str = "pspnet"
        manet: str = "manet"
        unet: str = "unet"
        unetv2: str = "unet++"
        tp_models: Tuple[str] = (
            deeplabv3,
            deeplabv4,
            fpn,
            linknet,
            manet,
            pan,
            pspnet,
            unet,
            unetv2
        )

        class Encoders:
            densenet121: str = "densenet121"
            densenet201: str = "densenet201"
            dpn68: str = "dpn68"
            efficientb0: str = "efficientnet-b0"
            efficientb1: str = "efficientnet-b1"
            efficientb2: str = "efficientnet-b2"
            efficientb3: str = "efficientnet-b3"
            efficientb4: str = "efficientnet-b4"
            efficientb5: str = "efficientnet-b5"
            efficientb6: str = "efficientnet-b6"
            efficientb7: str = "efficientnet-b7"
            resnet18: str = "resnet18"
            resnet50: str = "resnet50"
            resnext50: str = "resnext50_32x4d"
            seresnet50: str = "se_resnet50"
            vgg11bn: str = "vgg11_bn"
            xception: str = "xception"
            tp_encoders: Tuple[str] = (
                densenet121,
                densenet201,
                dpn68,
                efficientb0,
                efficientb1,
                efficientb2,
                efficientb3,
                efficientb4,
                efficientb5,
                efficientb6,
                efficientb7,
                resnet18,
                resnet50,
                resnext50,
                seresnet50,
                vgg11bn,
                xception,
            )

        class Optimisers:
            adam: str = "adam"
            tp_optimisers: Tuple[str] = (adam,)

        class Losses:
            bcelogits: str = "bcelogits"
            dicelogits: str = "dicelogits"
            focal: str = "focal"
            jaccard: str = "jaccard"
            tversky: str = "tversky"
            mse: str = "mse"
            tp_losses: Tuple[str] = (bcelogits, dicelogits, focal, jaccard, tversky, mse)
            str_multiclass: str = "multiclass"
            str_binary: str = "binary"
            str_multilabel: str = "multilabel"
            tp_modes: Tuple[str] = (str_binary, str_multiclass, str_multilabel)

    class Structures:
        clival_recess: str = "clival_recess"
        left_carotid: str = "left_carotid"
        left_optic_carotid_recess: str = "left_optic_carotid_recess"
        left_optic_protuberance: str = "left_optic_protuberance"
        planum_sphenoidal: str = "planum_sphenoidal"
        right_carotid: str = "right_carotid"
        right_optic_carotid_recess: str = "right_optic_carotid_recess"
        right_optic_protuberance: str = "right_optic_protuberance"
        sella: str = "sella"
        tuberculum_sellae: str = "tuberculum_sellae"
        tp_structures: Tuple[str] = (
            clival_recess,
            left_carotid,
            left_optic_carotid_recess,
            left_optic_protuberance,
            planum_sphenoidal,
            right_carotid,
            right_optic_carotid_recess,
            right_optic_protuberance,
            sella,
            tuberculum_sellae,
        )
        tp_structures_mrk: Tuple[str] = (
            left_carotid,
            left_optic_carotid_recess,
            left_optic_protuberance,
            planum_sphenoidal,
            right_carotid,
            right_optic_carotid_recess,
            right_optic_protuberance,
            tuberculum_sellae,
        )
        tp_structures_msk: Tuple[str] = (
            clival_recess,
            sella,
        )
        not_sella: str = "not_sella"
        tp_structures_all: str = tp_structures + (not_sella,)
        dt_colourmap: Dict[str, str] = {
            clival_recess: "yellow",
            left_carotid: "darkviolet",
            left_optic_carotid_recess: "orange",
            left_optic_protuberance: "lime",
            planum_sphenoidal: "slategrey",
            right_carotid: "magenta",
            right_optic_carotid_recess: "gold",
            right_optic_protuberance: "turquoise",
            sella: "blue",
            tuberculum_sellae: "green",
        }


class Log:
    """basic log functions to see outputs on terminal"""
    @staticmethod
    def arguments_pre():
        print("+" * 20)
        print("Passing arguments...")

    @staticmethod
    def arguments_post(args):
        [print(f"{arg}: {args[arg]}") for arg in args]
        print("Arguments passed...")
        print("+" * 20)

    @staticmethod
    def invalid(str_variable: str, ls_var_choices: List[str]):
        return f"{str_variable} is not a valid choice, valid choices are {ls_var_choices}."

    @staticmethod
    def console(str_m: str, bool_p: bool = True):
        if bool_p:
            print(str_m)


def device_initialise():
    """initialise device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise Exception("CUDA is not available")
    return device


def str2bool(str_value: str) -> bool:
    """manual check for {str_value} ensuring it is a bool argument"""
    if isinstance(str_value, bool):
        return str_value
    if str_value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str_value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2none(str_value: str):
    """manual check for {str_value} mapping to None if blank"""
    if str_value.lower() in ["", "none", "n/a"]:
        return None
    else:
        return str_value


class Directory:
    def __init__(self, path):
        self.path = path

    def create(self, bool_check=True):
        """create a directory, if {bool_check}=False, then do not check it already exists"""
        if bool_check:
            if not os.path.exists(self.path):
                os.mkdir(self.path)
        else:
            os.mkdir(self.path)

    def delete(self, bool_check=True):
        """delete a directory, if {bool_check}=False, then do not check it is empty"""
        if bool_check:
            os.rmdir(self.path)
        else:
            shutil.rmtree(self.path)


class PdUtils:
    def __init__(self, df):
        self.df = df

    def value(self, column: str):
        return self.df[column].to_list()[0]

    def ls_values(self, column: str):
        return self.df[column].to_list()

    def ls_values_unique(self, column: str):
        return self.df[column].unique().tolist()

    def subset(self, column: str, value) -> DataFrame:
        return self.df[self.df[column] == value]


def args_pop(args: Dict, tp_str_pop: Tuple) -> Dict:
    """delete arguments from args"""
    for str_pop in tp_str_pop:
        args.pop(str_pop)
    return args


def string_coordinate_to_float_coordinate(str_coordinates: str) -> List[Tuple[float, float]]:
    """convert {str_coordinates} into float coordinates"""
    ls_tp_coordinates: List[Tuple[float, float]] = []
    ls_coordinates: List[str] = re.split("[()]", str_coordinates)[1::2]
    for str_coordinate_euclid in ls_coordinates:
        x: float = float(re.split("[=,]", str_coordinate_euclid)[1::2][0])
        y: float = float(re.split("[=,]", str_coordinate_euclid)[1::2][1])
        ls_tp_coordinates.append((x, y))
    ls_tp_coordinates.append(ls_tp_coordinates[0])
    return ls_tp_coordinates


if __name__ == "__main__":
    main()
