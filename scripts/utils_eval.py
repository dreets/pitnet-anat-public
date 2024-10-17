# global imports
import os
import numpy as np
import pandas as pd
import torch
import warnings
from ast import literal_eval
from pathlib import Path
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

# local imports
from utils_main import device_initialise
from utils_main import Paths
from utils_main import Terms
from utils_main import PdUtils
from utils_modl import model_load
from utils_modl import dataloader_load

# strongly typed
from numpy import ndarray
from pandas.core.frame import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict
from typing import List
from typing import Tuple


warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)  # suppress precision warnings


def main():
    pass


def model_find_number(
        csv_model_map: Path = Paths.csv_model_map_msk,
        flt_model_number: str = Terms.Headings.flt_model_number,
) -> int:
    """find the next {str_model_number} to save down the model uniquely to {csv_model_map}"""
    df_model_map: DataFrame = pd.read_csv(csv_model_map)
    int_model_map_num_max: float = np.floor(df_model_map[flt_model_number].max())
    int_model_number: int = 1 if pd.isna(int_model_map_num_max) else int_model_map_num_max + 1
    return int(int_model_number)


def model_save_parameters(dt_parameters: Dict, csv_model_map: Path = Paths.csv_model_map_msk):
    """save {dt_parameters} to {csv_model_map}"""
    df_model_parameters: DataFrame = pd.DataFrame([dt_parameters])
    df_model_parameters.to_csv(csv_model_map, mode="a", header=False, index=False)
    print(f"Model parameters have been saved to {csv_model_map}.")


def model_save(dt_model: Dict[str, Tensor], str_model_number: str, path_models: Path = Paths.path_models):
    """save {dt_model} with {int_model_number} to {path_models}"""
    path_model = os.path.join(path_models, f"{str_model_number}.pth")
    torch.save(dt_model, path_model)
    print(f"Model has been saved to {path_model}.")


def model_save_history(
        ls_history: List[Dict[str, float]],
        int_model_number: int,
        path_history: Path = Paths.path_history,
):
    """save {ls_history} for to {path_history}/{int_model_number}.csv"""
    df_history: DataFrame = pd.DataFrame(ls_history)
    path_model_history = os.path.join(path_history, f"{int_model_number}.csv")
    df_history.to_csv(path_model_history, index=False)
    print(f"Model history has been saved to {path_model_history}.")


def model_find_metrics(
        model,
        str_mode: str,
        tp_structures: Tuple[str],
        dt_loaders: Dict[str, DataLoader],
        device,
        tp_split: Tuple[str] = Terms.tp_split,
        tp_metrics: Tuple[str] = Terms.Metrics.tp_metrics,
) -> Dict[str, float]:
    """find the (acc, dice, and iou) of {model} given data from {dt_loaders}"""
    dt_metrics: Dict[str, float] = {}
    for str_split in tp_split:
        for str_structure in tp_structures:
            for str_metric in tp_metrics:
                dt_metrics[f"{str_structure}_{str_split}_{str_metric}"] = 0

        loader: DataLoader = dt_loaders[str_split]
        model.eval()

        with torch.no_grad():
            for img, msks in tqdm(loader):
                ts_img: Tensor = img.float().to(device)
                ts_msks: Tensor = msks.float().to(device)
                ts_prds: Tensor = torch.sigmoid(model(ts_img))
                ts_prds: Tensor = (ts_prds > 0.5).float()

                for int_class, str_structure in enumerate(tp_structures):
                    ts_prd: Tensor = ts_prds[:, int_class, ...]
                    if str_mode in (Terms.Models.Losses.str_binary, Terms.Models.Losses.str_multilabel):
                        ts_msk = ts_msks[:, int_class, ...]
                    else:
                        ts_msk = (ts_msks == int_class).float()

                    dt_metrics_temp: Dict[str, float] = metrics_calculate(ts_msk=ts_msk, ts_prd=ts_prd)
                    for str_metric in tp_metrics:
                        dt_metrics[f"{str_structure}_{str_split}_{str_metric}"] += dt_metrics_temp[str_metric]

            for str_structure in tp_structures:
                for str_metric in tp_metrics:
                    dt_metrics[f"{str_structure}_{str_split}_{str_metric}"] *= 100 / len(loader)
    return dt_metrics


def metrics_calculate_fundamental(ts_act: Tensor, ts_prd: Tensor) -> Tuple[int, int, int, int]:
    """calculate basic metrics (true positive, true negative, false positive, false negative)"""
    ts_confusion: Tensor = ts_prd / ts_act
    int_tp: int = torch.sum(ts_confusion == 1).item()
    int_tn: int = torch.sum(torch.isnan(ts_confusion)).item()
    int_fp: int = torch.sum(ts_confusion == float("inf")).item()
    int_fn: int = torch.sum(ts_confusion == 0).item()
    tp_fundamental_metrics: Tuple[int, int, int, int] = (int_tp, int_tn, int_fp, int_fn)
    return tp_fundamental_metrics


def metrics_calculate(ts_msk: Tensor, ts_prd: Tensor) -> Dict[str, float]:
    """calculate evaluation metrics (accuracy, recall, precision, dice, iou)"""
    dt_metrics: Dict[str, float] = {}
    flt_smooth: float = 0.000001
    int_tp, int_tn, int_fp, int_fn = metrics_calculate_fundamental(ts_act=ts_msk, ts_prd=ts_prd)
    int_all: int = int_tp + int_tn + int_fp + int_fn
    dt_metrics[Terms.Metrics.acc] = (int_tp + int_tn + flt_smooth) / (int_all + flt_smooth)
    dt_metrics[Terms.Metrics.rec] = (int_tp + flt_smooth) / (int_tp + int_fn + flt_smooth)
    dt_metrics[Terms.Metrics.pre] = (int_tp + flt_smooth) / (int_tp + int_fp + flt_smooth)

    flt_inter: float = float((ts_prd * ts_msk).sum())
    flt_union: float = float(ts_prd.sum() + ts_msk.sum()) - flt_inter
    flt_iou: float = (flt_inter + flt_smooth) / (flt_union + flt_smooth)
    dt_metrics[Terms.Metrics.dce] = 2 * flt_iou / (1 + flt_iou)
    dt_metrics[Terms.Metrics.iou] = flt_iou
    return dt_metrics


def epoch_find_best_from_history(
        int_model: int,
        path_history: str = Paths.path_history,
        csv_model_map: str = Paths.csv_model_map_msk,
) -> Dict[str, int]:
    """back calculate the best epoch for {int_model} from {path_history}"""
    df_history: DataFrame = pd.read_csv(os.path.join(path_history, f"{int_model}.csv"))
    df_model_map: DataFrame = pd.read_csv(csv_model_map)
    df_model: DataFrame = df_model_map[df_model_map[Terms.Headings.flt_model_number] == int_model]
    str_tp_str_structures: str = df_model[Terms.Headings.tp_str_structures].to_list()[0]
    tp_str_structures: Tuple[str] = literal_eval(str_tp_str_structures)

    dt_epochs: Dict[str, int] = {}
    for str_structure in tp_str_structures:
        str_key: str = f"{str_structure}_{Terms.val}_{Terms.Metrics.iou}"
        df_metric_max: DataFrame = df_history[df_history[str_key] == df_history[str_key].max()]
        int_epoch: int = df_metric_max[Terms.Headings.int_epoch].to_list()[0]
        dt_epochs[str_structure] = int_epoch
    return dt_epochs


def metrics_from_model(
        flt_model_number: float,
        tp_split: Tuple = Terms.tp_split,
        csv_model_map: str = Paths.csv_model_map_msk,
) -> Dict[str, float]:
    """calculate evaluation metrics for {flt_model_number} using a forward pass"""
    df_model_map: DataFrame = pd.read_csv(csv_model_map)
    df_model: DataFrame = df_model_map[df_model_map[Terms.Headings.flt_model_number] == flt_model_number]
    dt_metrics: Dict[str, float] = model_find_metrics(
        model=model_load(flt_model_number=flt_model_number),
        str_mode=PdUtils(df_model).value("str_mode"),
        tp_structures=literal_eval(PdUtils(df_model).value(Terms.Headings.tp_str_structures)),
        dt_loaders=dataloader_load(flt_model_number=flt_model_number),
        device=device_initialise(),
        tp_split=tp_split,
    )
    return dt_metrics


def calculate_structure_metrics(dt_ts_ts: Dict[str, Tuple[Tensor, Tensor]]):
    """calculate evaluation metrics given a ground truth and prediction dictionary of tensors={dt_ts_ts}"""
    dt_metrics: Dict[str, float] = {}
    for str_structure in dt_ts_ts.keys():
        ts_msk, ts_prd = dt_ts_ts[str_structure]
        dt_structure_metrics: Dict[str, float] = metrics_calculate(ts_msk=ts_msk, ts_prd=ts_prd)
        for str_metric in Terms.Metrics.tp_metrics:
            dt_metrics[f"{str_structure}_{str_metric}"] = dt_structure_metrics[str_metric]
    return dt_metrics


def find_batch_pck(
        ts_mrks: Tensor,
        ts_prds: Tensor,
        ts_msks: Tensor,
        loader: DataLoader,
        tp_thresholds: Tuple[float] = Terms.Metrics.tp_thresholds,
        tp_structures: Tuple[str] = Terms.Structures.tp_structures,
) -> ndarray:
    """calculate the PCKs given a prediction={ts_prds} and ground truth={ts_mrks}"""
    ay_metrics_batch: ndarray = np.zeros([len(tp_structures), len(tp_thresholds) + 1])
    ay_classes: ndarray = np.zeros([len(tp_structures)])
    for int_i in range(len(ts_mrks)):  # batch_size
        ay_metrics_sum: ndarray = np.zeros([len(tp_structures), len(tp_thresholds) + 1])
        for int_j in range(len(ts_mrks[0])):
            if not (int_j % 2):
                int_structure: int = int(int_j / 2)
                x_prd = ts_prds[int_i][int_j].item()
                y_prd = ts_prds[int_i][int_j + 1].item()
                x_mrk = ts_mrks[int_i][int_j].item()
                y_mrk = ts_mrks[int_i][int_j + 1].item()
                if x_mrk and y_mrk:
                    ay_classes[int_structure] += 1
                    ay_metrics_sum[int_structure][:-1] += np.array(ls_keypoints(
                        x_prd=x_prd,
                        y_prd=y_prd,
                        x_mrk=x_mrk,
                        y_mrk=y_mrk,
                        tp_thresholds=tp_thresholds,
                    ))
                    ay_metrics_sum[int_structure][-1] += int(loader.dataset.bool_prediction_in_mask(
                        x_prd=x_prd,
                        y_prd=y_prd,
                        ts_msk=ts_msks[int_i].permute(2, 0, 1)[int_structure],
                    ))
        ay_metrics_batch += ay_metrics_sum

    for int_structure, int_classes in enumerate(ay_classes):
        if int_classes:
            ay_metrics_batch[int_structure] /= int_classes

    return ay_metrics_batch


def flt_squared_error(x_prd: float, y_prd: float, x_mrk: float, y_mrk: float) -> float:
    """calculate the euclidean error between a prediction=({x_prd},{y_prd}) and ground_truth=({x_mrk},{y_mrk})"""
    return np.sqrt((16 / 9) ** 2 * (x_prd - x_mrk) ** 2 + (y_prd - y_mrk) ** 2)


def ls_keypoints(x_prd: float, y_prd: float, x_mrk: float, y_mrk: float, tp_thresholds: Tuple[float]) -> List[int]:
    """is the prediction=({x_prd},{y_prd}) within a radius={tp_thresholds} of the ground_truth=({x_mrk},{y_mrk})"""
    flt_euclidean = flt_squared_error(x_prd=x_prd, y_prd=y_prd, x_mrk=x_mrk, y_mrk=y_mrk)
    ls_counters: List[int] = []
    for flt_threshold in tp_thresholds:
        if flt_euclidean <= flt_threshold:
            ls_counters.append(1)
        else:
            ls_counters.append(0)
    return ls_counters


def find_batch_iou(
    ts_msks: Tensor,
    ts_prds: Tensor,
    tp_structures: Tuple[str],
    tp_structures_msk: Tuple[str],
    tp_metrics: Tuple[str] = Terms.Metrics.tp_metrics,
    str_mode: str = Terms.Models.Losses.str_multiclass,
    str_type: str = "multi",
) -> ndarray:
    """calculate the iou for a batch of predictions={ts_prds} given the ground_truth={ts_msks}"""
    ay_metrics: ndarray = np.zeros([len(tp_structures_msk), len(tp_metrics)])
    for int_i in range(len(ts_msks)):
        int_s: int = 0

        for int_structure, str_structure in enumerate(tp_structures):
            if str_structure in tp_structures_msk:
                ts_prd_class: Tensor = ts_prds[:, int_structure, ...].float()
                if str_mode in (Terms.Models.Losses.str_binary, Terms.Models.Losses.str_multilabel):
                    ts_msk_class = ts_msks[:, int_structure, ...]
                else:
                    if str_type == "multi":
                        ts_msk_class: Tensor = ts_msks[..., int_structure].float()
                    else:
                        ts_msk_class: Tensor = (ts_msks == int_structure).long()
                ay_metric = np.array(list(metrics_calculate(ts_msk=ts_msk_class, ts_prd=ts_prd_class).values()))
                ay_metrics[int_s] += ay_metric
                int_s += 1
    ay_metrics /= len(ts_msks)
    return ay_metrics


if __name__ == "__main__":
    main()
