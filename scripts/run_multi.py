# global imports
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# local imports
from utils_eval import find_batch_pck
from utils_eval import find_batch_iou
from utils_eval import model_find_number
from utils_eval import model_save
from utils_eval import model_save_history
from utils_eval import model_save_parameters
from utils_main import Log
from utils_main import Paths
from utils_main import Terms
from utils_main import device_initialise
from utils_main import args_pop
from utils_mark import dt_loaders_initialise
from utils_mark import loss_initialise as loss_initialise_mrk
from utils_mark import model_initialise
from utils_mark import optimiser_initialise
from utils_mark import find_tensor_markers
from utils_modl import loss_initialise as loss_initialise_msk

# strongly typed
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict
from typing import List
from typing import Tuple


def main():
    """create & train the neural network"""
    # passing arguments
    Log.arguments_pre()
    parser = argparse.ArgumentParser(description="choose parameters")

    # data variables
    parser.add_argument("--int_fold", type=int, default=0)
    parser.add_argument("--tp_structures_mrk", nargs="+", type=str,
                        choices=Terms.Structures.tp_structures, default=Terms.Structures.tp_structures_mrk)
    parser.add_argument("--tp_structures_msk", nargs="+", type=str,
                        choices=Terms.Structures.tp_structures, default=Terms.Structures.tp_structures_msk)
    parser.add_argument("--tp_thresholds", nargs="+", type=float, default=Terms.Metrics.tp_thresholds)
    parser.add_argument("--tp_metrics", nargs="+", type=float, default=Terms.Metrics.tp_metrics)
    parser.add_argument("--str_model", type=str, default=Terms.Models.unetv2)

    # training variables
    parser.add_argument("--flt_rate_mrk", type=float, default=0.0001)
    parser.add_argument("--flt_rate_msk", type=float, default=0.0001)
    parser.add_argument("--int_batch", type=int, default=6)
    parser.add_argument("--int_epochs", type=int, default=20)
    parser.add_argument("--str_optimiser_mrk", type=str,
                        choices=Terms.Models.Optimisers.tp_optimisers, default=Terms.Models.Optimisers.adam)
    parser.add_argument("--str_optimiser_msk", type=str,
                        choices=Terms.Models.Optimisers.tp_optimisers, default=Terms.Models.Optimisers.adam)
    parser.add_argument("--str_loss_mrk", type=str,
                        choices=Terms.Models.Losses.tp_losses, default=Terms.Models.Losses.mse)
    parser.add_argument("--str_loss_msk", type=str,
                        choices=Terms.Models.Losses.tp_losses, default=Terms.Models.Losses.focal)

    # defaults (not necessary to change)
    parser.add_argument("--int_height", type=int, default=736)
    parser.add_argument("--int_width", type=int, default=1280)

    # paths (not necessary to change)
    parser.add_argument("--csv_coordinates", type=str, default=Paths.csv_coordinates)
    parser.add_argument("--csv_model_map_mul", type=str, default=Paths.csv_model_map_mul)
    parser.add_argument("--csv_split", type=str, default=Paths.csv_split)
    parser.add_argument("--path_images", type=str, default=Paths.path_input_images)
    parser.add_argument("--path_masks", type=str, default=Paths.path_input_masks)
    parser.add_argument("--path_models_mul", type=str, default=Paths.path_models_mul)
    parser.add_argument("--path_history_mul", type=str, default=Paths.path_history_mul)

    # arguments passed
    args = vars(parser.parse_args())
    Log.arguments_post(args)

    # args to variables
    csv_coordinates: Path = Path(args["csv_coordinates"])
    csv_model_map_mul: Path = Path(args["csv_model_map_mul"])
    csv_split: Path = Path(args["csv_split"])
    int_batch: int = int(args["int_batch"])
    int_epochs: int = int(args["int_epochs"])
    int_fold: int = int(args["int_fold"])
    int_height: int = int(args["int_height"])
    int_width: int = int(args["int_width"])
    path_history_mul: Path = Path(args["path_history_mul"])
    path_images: Path = Path(args["path_images"])
    path_masks: Path = Path(args["path_masks"])
    path_models_mul: Path = Path(args["path_models_mul"])
    str_loss_mrk: str = str(args["str_loss_mrk"])
    str_loss_msk: str = str(args["str_loss_msk"])
    str_optimiser_mrk: str = str(args["str_optimiser_mrk"])
    str_optimiser_msk: str = str(args["str_optimiser_msk"])
    str_model: str = str(args["str_model"])
    tp_metrics: Tuple[str] = tuple(args["tp_metrics"])
    tp_structures_mrk: Tuple[str] = tuple(args["tp_structures_mrk"])
    tp_structures_msk: Tuple[str] = tuple(args["tp_structures_msk"])
    tp_thresholds: Tuple[float] = tuple(args["tp_thresholds"])
    flt_rate_mrk: float = float(args["flt_rate_mrk"])
    flt_rate_msk: float = float(args["flt_rate_msk"])
    device = device_initialise()

    # initialise
    model = model_initialise(
        str_model=str_model,
        int_classes=len(tp_structures_mrk) + len(tp_structures_msk),
        device=device,
    )
    optimiser_mrk = optimiser_initialise(
        str_optimiser=str_optimiser_mrk,
        model_parameters=model.parameters(),
        flt_rate=flt_rate_mrk,
    )
    optimiser_msk = optimiser_initialise(
        str_optimiser=str_optimiser_msk,
        model_parameters=model.parameters(),
        flt_rate=flt_rate_msk,
    )
    loss_mrk = loss_initialise_mrk(
        str_loss=str_loss_mrk,
    )
    loss_msk = loss_initialise_msk(
        str_loss=str_loss_msk,
        str_mode="multiclass",
        str_type="multi",
    )
    dt_loaders = dt_loaders_initialise(
        int_fold=int_fold,
        int_height=int_height,
        int_width=int_width,
        int_batch=int_batch,
        tp_structures_mrk=tp_structures_mrk,
        tp_structures_msk=tp_structures_msk,
        path_images=path_images,
        path_masks=path_masks,
        csv_split=csv_split,
        csv_coordinates=csv_coordinates,
    )

    # train
    model_best, ls_history, int_epoch_best, flt_best_metric = model_train(
        int_epochs=int_epochs,
        dt_loaders=dt_loaders,
        model=model,
        optimiser_mrk=optimiser_mrk,
        optimiser_msk=optimiser_msk,
        loss_mrk=loss_mrk,
        loss_msk=loss_msk,
        device=device,
        tp_structures_mrk=tp_structures_mrk,
        tp_structures_msk=tp_structures_msk,
        tp_thresholds=tp_thresholds,
        tp_metrics=tp_metrics,
    )

    # save
    int_model_number: int = model_find_number(csv_model_map=csv_model_map_mul)
    model_save_history(ls_history=ls_history, int_model_number=int_model_number, path_history=path_history_mul)
    model_save(dt_model=model_best.state_dict(), str_model_number=str(int_model_number), path_models=path_models_mul)

    dt_parameters: Dict = {"flt_model_number": int_model_number}
    dt_parameters.update(args_pop(
        args=args,
        tp_str_pop=(
            "csv_coordinates",
            "csv_model_map_mul",
            "csv_split",
            "path_history_mul",
            "path_images",
            "path_masks",
            "path_models_mul",
        )
    ))
    dt_parameters["int_epoch"] = int_epoch_best
    for str_split in Terms.tp_split:
        for int_a, flt_threshold in enumerate(tp_thresholds):
            dt_parameters[f"{str_split}_pck_{flt_threshold}"] = ls_history[int_epoch_best][f"{str_split}_pck"].T[int_a]
        dt_parameters[f"{str_split}_in_mask"] = ls_history[int_epoch_best][f"{str_split}_pck"].T[-1]
        for int_b, flt_metric in enumerate(tp_metrics):
            dt_parameters[f"{str_split}_{flt_metric}"] = ls_history[int_epoch_best][f"{str_split}_iou"].T[int_b]
    dt_parameters["best_metric"] = flt_best_metric
    model_save_parameters(dt_parameters=dt_parameters, csv_model_map=csv_model_map_mul)

    Log.console(f"The best model has metric={flt_best_metric} at epoch={int_epoch_best}.")


def model_train(
        int_epochs: int,
        dt_loaders: Dict[str, DataLoader],
        model,
        optimiser_mrk,
        optimiser_msk,
        loss_mrk,
        loss_msk,
        device,
        tp_structures_mrk: Tuple[str] = Terms.Structures.tp_structures_mrk,
        tp_structures_msk: Tuple[str] = Terms.Structures.tp_structures_msk,
        tp_thresholds: Tuple[float] = Terms.Metrics.tp_thresholds,
        tp_metrics: Tuple[str] = Terms.Metrics.tp_metrics,
) -> Tuple:
    """train the {model} and find the optimal model based on {str_metric}"""
    ls_history: List[Dict[str, float]] = []
    model_best = model
    flt_metric_best: float = 0
    int_epoch_best: int = 0
    for int_epoch in range(int_epochs):
        Log.console("-" * 20)
        Log.console(f"Epoch: {int_epoch}")
        model, dt_metrics = model_epoch(
            dt_loaders=dt_loaders,
            model=model,
            optimiser_mrk=optimiser_mrk,
            optimiser_msk=optimiser_msk,
            loss_mrk=loss_mrk,
            loss_msk=loss_msk,
            device=device,
            tp_thresholds=tp_thresholds,
            tp_metrics=tp_metrics,
            tp_structures_mrk=tp_structures_mrk,
            tp_structures_msk=tp_structures_msk,
        )
        Log.console(str(dt_metrics))

        flt_err: float = float(np.mean(dt_metrics["val_pck"].T[4]))
        flt_metric: float = flt_err

        if flt_metric > flt_metric_best:
            flt_metric_best = flt_metric
            model_best = model
            int_epoch_best = int_epoch

        dt_history: Dict = {Terms.Headings.int_epoch: int_epoch}
        dt_history.update(dt_metrics)
        ls_history.append(dt_history)

    tp_model_history: Tuple = (model_best, ls_history, int_epoch_best, flt_metric_best)
    return tp_model_history


def model_epoch(
        dt_loaders: Dict[str, DataLoader],
        model,
        optimiser_mrk,
        optimiser_msk,
        loss_mrk,
        loss_msk,
        tp_structures_mrk: Tuple[str],
        tp_structures_msk: Tuple[str],
        tp_thresholds: Tuple[float] = Terms.Metrics.tp_thresholds,
        tp_metrics: Tuple[str] = Terms.Metrics.tp_metrics,
        tp_split: Tuple[str] = Terms.tp_split,
        device=device_initialise(),
) -> Tuple:
    """train the model by one epoch, returning the model"""
    dt_metrics = {}
    for str_split in tp_split:
        if str_split == Terms.train:
            model.train()
        else:
            model.eval()

        ay_pck_total: ndarray = np.zeros([len(tp_structures_mrk), len(tp_thresholds) + 1])
        ay_iou_total: ndarray = np.zeros([len(tp_structures_msk), len(tp_metrics)])
        loader = dt_loaders[str_split]
        loop = tqdm(loader)
        ts_loss = 0
        int_number: int = 0
        if len(loader) - int_number != 1:
            for imgs, mrks, msks in loop:
                with torch.set_grad_enabled(str_split == Terms.train):
                    # marks first
                    ts_imgs, ts_mrks, ts_msks, ts_prds_mrks, ts_prds_msks = find_tensor_markers(
                        imgs=imgs,
                        mrks=mrks,
                        msks=msks,
                        model=model,
                        loader=loader,
                        device=device,
                    )
                    ts_lss_msk: Tensor = loss_msk(ts_prds_msks.to(device=device), ts_msks.to(device=device))

                    if str_split == Terms.train:
                        optimiser_msk.zero_grad()
                        ts_lss_msk.backward()
                        optimiser_msk.step()

                    # masks second
                    ts_imgs, ts_mrks, ts_msks, ts_prds_mrks, ts_prds_msks = find_tensor_markers(
                        imgs=imgs,
                        mrks=mrks,
                        msks=msks,
                        model=model,
                        loader=loader,
                        device=device,
                    )
                    ts_lss_mrk: Tensor = loss_mrk(ts_prds_mrks.to(device=device), ts_mrks.to(device=device))

                    if str_split == Terms.train:
                        optimiser_mrk.zero_grad()
                        ts_lss_mrk.backward()
                        optimiser_mrk.step()

                    ts_loss += (ts_lss_mrk + ts_lss_msk) / 2

                # update loop
                loop.set_postfix(loss=ts_loss.item())

                # update metrics
                ay_pck_total += find_batch_pck(
                    ts_mrks=ts_mrks,
                    ts_prds=ts_prds_mrks,
                    ts_msks=ts_msks,
                    loader=loader,
                    tp_structures=tp_structures_mrk,
                    tp_thresholds=tp_thresholds,
                )
                ay_iou_total += find_batch_iou(
                    ts_msks=ts_msks,
                    ts_prds=(torch.sigmoid(ts_prds_msks) > 0.5).float(),
                    tp_structures=tp_structures_mrk + tp_structures_msk,
                    tp_structures_msk=tp_structures_msk,
                    tp_metrics=tp_metrics,
                )
                int_number += len(imgs)

            ay_pck_total = ay_pck_total / len(loader) * 100
            ay_iou_total = ay_iou_total / len(loader) * 100

            dt_metrics[f"{str_split}_pck"] = ay_pck_total
            dt_metrics[f"{str_split}_iou"] = ay_iou_total

    return model, dt_metrics


if __name__ == "__main__":
    main()
