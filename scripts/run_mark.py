# global imports
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# local imports
from utils_eval import find_batch_pck
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
from utils_mark import loss_initialise
from utils_mark import model_initialise
from utils_mark import optimiser_initialise
from utils_mark import find_tensor_markers

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
    parser.add_argument("--tp_structures", nargs="+", type=str,
                        choices=Terms.Structures.tp_structures, default=Terms.Structures.tp_structures_mrk)
    parser.add_argument("--tp_thresholds", nargs="+", type=float, default=Terms.Metrics.tp_thresholds)
    parser.add_argument("--str_model", type=str, default=Terms.Models.unetv2)

    # training variables
    parser.add_argument("--flt_rate", type=float, default=0.0001)
    parser.add_argument("--int_batch", type=int, default=6)
    parser.add_argument("--int_epochs", type=int, default=20)
    parser.add_argument("--str_optimiser", type=str,
                        choices=Terms.Models.Optimisers.tp_optimisers, default=Terms.Models.Optimisers.adam)
    parser.add_argument("--str_loss", type=str,
                        choices=Terms.Models.Losses.tp_losses, default=Terms.Models.Losses.mse)

    # defaults (not necessary to change)
    parser.add_argument("--int_height", type=int, default=736)
    parser.add_argument("--int_width", type=int, default=1280)

    # paths (not necessary to change)
    parser.add_argument("--csv_coordinates", type=str, default=Paths.csv_coordinates)
    parser.add_argument("--csv_model_map_marks", type=str, default=Paths.csv_model_map_mrk)
    parser.add_argument("--csv_split", type=str, default=Paths.csv_split)
    parser.add_argument("--path_history_marks", type=str, default=Paths.path_history_mrk)
    parser.add_argument("--path_images", type=str, default=Paths.path_input_images)
    parser.add_argument("--path_masks", type=str, default=Paths.path_input_masks)
    parser.add_argument("--path_models_marks", type=str, default=Paths.path_models_mrk)

    # arguments passed
    args = vars(parser.parse_args())
    Log.arguments_post(args)

    # args to variables
    csv_coordinates: Path = Path(args["csv_coordinates"])
    csv_model_map_marks: Path = Path(args["csv_model_map_marks"])
    csv_split: Path = Path(args["csv_split"])
    int_batch: int = int(args["int_batch"])
    int_epochs: int = int(args["int_epochs"])
    int_fold: int = int(args["int_fold"])
    int_height: int = int(args["int_height"])
    int_width: int = int(args["int_width"])
    path_images: Path = Path(args["path_images"])
    path_history_marks: Path = Path(args["path_history_marks"])
    path_masks: Path = Path(args["path_masks"])
    path_models_marks: Path = Path(args["path_models_marks"])
    str_loss: str = str(args["str_loss"])
    str_optimiser: str = str(args["str_optimiser"])
    str_model: str = str(args["str_model"])
    tp_structures: Tuple[str] = tuple(args["tp_structures"])
    tp_thresholds: Tuple[float] = tuple(args["tp_thresholds"])
    flt_rate: float = float(args["flt_rate"])
    device = device_initialise()

    # initialise
    model = model_initialise(
        str_model=str_model,
        int_classes=len(tp_structures),
        device=device,
    )
    optimiser = optimiser_initialise(
        str_optimiser=str_optimiser,
        model_parameters=model.parameters(),
        flt_rate=flt_rate,
    )
    loss = loss_initialise(
        str_loss=str_loss,
    )
    dt_loaders = dt_loaders_initialise(
        int_fold=int_fold,
        int_height=int_height,
        int_width=int_width,
        int_batch=int_batch,
        tp_structures_mrk=tp_structures,
        tp_structures_msk=(),
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
        optimiser=optimiser,
        loss=loss,
        device=device,
        tp_structures=tp_structures,
        tp_thresholds=tp_thresholds,
    )

    # save
    int_model_number: int = model_find_number(csv_model_map=csv_model_map_marks)
    model_save_history(ls_history=ls_history, int_model_number=int_model_number, path_history=path_history_marks)
    model_save(dt_model=model_best.state_dict(), str_model_number=str(int_model_number), path_models=path_models_marks)

    dt_parameters: Dict = {"flt_model_number": int_model_number}
    dt_parameters.update(args_pop(
        args=args,
        tp_str_pop=(
            "csv_coordinates",
            "csv_model_map_marks",
            "csv_split",
            "path_history_marks",
            "path_images",
            "path_masks",
            "path_models_marks",
        )
    ))
    dt_parameters["int_epoch"] = int_epoch_best
    for int_index, flt_threshold in enumerate(tp_thresholds):
        dt_parameters[f"train_pck_{flt_threshold}"] = ls_history[int_epoch_best]["train"].T[int_index]
    dt_parameters[f"train_in_mask"] = ls_history[int_epoch_best]["train"].T[-1]
    for int_index, flt_threshold in enumerate(tp_thresholds):
        dt_parameters[f"val_pck_{flt_threshold}"] = ls_history[int_epoch_best]["val"].T[int_index]
    dt_parameters[f"val_in_mask"] = ls_history[int_epoch_best]["val"].T[-1]
    dt_parameters[f"val_metric"] = flt_best_metric
    model_save_parameters(dt_parameters=dt_parameters, csv_model_map=csv_model_map_marks)

    Log.console(f"The best model has values {ls_history[int_epoch_best]['val']}.")


def model_train(
        int_epochs: int,
        dt_loaders: Dict[str, DataLoader],
        model,
        optimiser,
        loss,
        device,
        tp_thresholds: Tuple[float] = Terms.Metrics.tp_thresholds,
        tp_structures: Tuple[str] = Terms.Structures.tp_structures,
) -> Tuple:
    """train the {model} and find the optimal model based on {str_metric}"""
    ls_history: List[Dict[str, float]] = []
    model_best = model
    flt_metric_best: int = 0
    int_epoch_best: int = 0
    for int_epoch in range(int_epochs):
        Log.console("-" * 20)
        Log.console(f"Epoch: {int_epoch}")
        model, dt_metrics = model_epoch(
            dt_loaders=dt_loaders,
            model=model,
            optimiser=optimiser,
            loss=loss,
            device=device,
            tp_thresholds=tp_thresholds,
            tp_structures=tp_structures,
        )
        Log.console(str(dt_metrics))

        flt_metric = 0
        for int_index, flt_threshold in enumerate(tp_thresholds):
            if flt_threshold == 0.20:
                flt_metric = np.mean(dt_metrics["val"].T[int_index])

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
        optimiser,
        loss,
        device=device_initialise(),
        tp_structures: Tuple[str] = Terms.Structures.tp_structures,
        tp_thresholds: Tuple[float] = Terms.Metrics.tp_thresholds,
        tp_split: Tuple[str] = Terms.tp_split,
) -> Tuple:
    """train the model by one epoch, returning the model"""
    dt_metrics: Dict = {}
    for str_split in tp_split:
        if str_split == Terms.train:
            model.train()
        else:
            model.eval()

        ay_error_total: ndarray = np.zeros([len(tp_structures), len(tp_thresholds) + 1])
        loader = dt_loaders[str_split]
        loop = tqdm(loader)
        try:
            for imgs, mrks, msks in loop:
                with torch.set_grad_enabled(str_split == Terms.train):
                    ts_imgs, ts_mrks, ts_msks, ts_prds, ts_prds_msks = find_tensor_markers(
                        imgs=imgs,
                        mrks=mrks,
                        msks=msks,
                        model=model,
                        loader=loader,
                        device=device,
                    )
                    ts_lss: Tensor = loss(ts_prds.to(device=device), ts_mrks.to(device=device))

                    # backward
                    if str_split == Terms.train:
                        model.zero_grad()
                        ts_lss.backward()
                        optimiser.step()

                # update loop
                loop.set_postfix(loss=ts_lss.item())

                # update metrics
                ay_error_total += find_batch_pck(
                    ts_mrks=ts_mrks,
                    ts_prds=ts_prds,
                    ts_msks=ts_msks,
                    loader=loader,
                    tp_structures=tp_structures,
                    tp_thresholds=tp_thresholds,
                )

            ay_error_total = ay_error_total / len(loader) * 100

            dt_metrics[f"{str_split}"] = ay_error_total

        except ValueError:
            Log.console("ValueError, due to final batch size equalling 1, this has been excluded from training.")
    return model, dt_metrics


if __name__ == "__main__":
    main()
