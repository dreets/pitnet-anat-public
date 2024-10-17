# global imports
import argparse
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm


# local imports
from utils_eval import model_find_number
from utils_eval import find_batch_iou
from utils_eval import model_save
from utils_eval import model_save_history
from utils_eval import model_save_parameters
from utils_main import device_initialise
from utils_main import Log
from utils_main import Paths
from utils_main import Terms
from utils_modl import dt_loaders_initialise
from utils_modl import loss_initialise
from utils_modl import model_initialise
from utils_modl import optimiser_initialise

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
    parser.add_argument("--str_structures", type=str, default=Terms.Structures.sella)  # this is deprecated but required
    parser.add_argument("--tp_structures", nargs="+", type=str,
                        choices=Terms.Structures.tp_structures, default=Terms.Structures.tp_structures_msk)

    # model variables
    parser.add_argument("--str_model", type=str, choices=Terms.Models.tp_models, default=Terms.Models.unetv2)
    parser.add_argument("--str_encoder", type=str,
                        choices=Terms.Models.Encoders.tp_encoders, default=Terms.Models.Encoders.efficientb3)
    parser.add_argument("--int_classes", type=int, default=1)
    parser.add_argument("--str_activation", type=str, default="none")

    # training variables
    parser.add_argument("--flt_rate", type=float, default=0.001)
    parser.add_argument("--int_batch", type=int, default=6)
    parser.add_argument("--int_epochs", type=int, default=20)
    parser.add_argument("--str_mode", type=str,
                        choices=Terms.Models.Losses.tp_modes, default=Terms.Models.Losses.str_multiclass)
    parser.add_argument("--str_metric", type=str, default=Terms.Metrics.iou)
    parser.add_argument("--str_optimiser", type=str,
                        choices=Terms.Models.Optimisers.tp_optimisers, default=Terms.Models.Optimisers.adam)
    parser.add_argument("--str_loss", type=str,
                        choices=Terms.Models.Losses.tp_losses, default=Terms.Models.Losses.focal)

    # defaults (not necessary to change)
    parser.add_argument("--int_channels", type=int, default=3)
    parser.add_argument("--int_height", type=int, default=736)
    parser.add_argument("--int_width", type=int, default=1280)

    # paths (not necessary to change)
    parser.add_argument("--csv_model_map", type=str, default=Paths.csv_model_map_msk)
    parser.add_argument("--csv_split", type=str, default=Paths.csv_split)
    parser.add_argument("--path_images", type=str, default=Paths.path_input_images)
    parser.add_argument("--path_history", type=str, default=Paths.path_history_msk)
    parser.add_argument("--path_masks", type=str, default=Paths.path_input_masks)
    parser.add_argument("--path_models", type=str, default=Paths.path_models_msk)

    # arguments passed
    args = vars(parser.parse_args())
    Log.arguments_post(args)

    # args to variables
    csv_model_map: Path = args["csv_model_map"]
    csv_split: Path = args["csv_split"]
    flt_rate: float = args["flt_rate"]
    int_batch: int = args["int_batch"]
    int_channels: int = args["int_channels"]
    int_epochs: int = args["int_epochs"]
    int_fold: int = args["int_fold"]
    int_height: int = args["int_height"]
    int_width: int = args["int_width"]
    path_images: Path = args["path_images"]
    path_masks: Path = args["path_masks"]
    path_models: Path = args["path_models"]
    path_history: Path = args["path_history"]
    str_activation: str = args["str_activation"]
    str_encoder: str = args["str_encoder"]
    str_loss: str = args["str_loss"]
    str_metric: str = args["str_metric"]
    str_mode: str = args["str_mode"]
    str_model: str = args["str_model"]
    str_optimiser: str = args["str_optimiser"]
    tp_structures: Tuple[str] = tuple(args["tp_structures"])
    int_classes: int = len(tp_structures)
    args["tp_structures"]: int = args["tp_structures"]
    args["int_classes"]: int = int_classes
    device = device_initialise()

    # initialise
    model = model_initialise(
        str_model=str_model,
        str_encoder=str_encoder,
        int_channels=int_channels,
        int_classes=int_classes,
        str_activation=str_activation,
        device=device,
    )
    optimiser = optimiser_initialise(
        str_optimiser=str_optimiser,
        model_parameters=model.parameters(),
        flt_rate=flt_rate,
    )
    loss = loss_initialise(str_loss=str_loss, str_mode=str_mode)
    dt_loaders: Dict[str, DataLoader] = dt_loaders_initialise(
        int_fold=int_fold,
        str_mode=str_mode,
        int_height=int_height,
        int_width=int_width,
        tp_structures=tp_structures,
        int_batch=int_batch,
        path_images=path_images,
        path_masks=path_masks,
        csv_split=csv_split,
    )

    # train
    tp_model_history: Tuple = model_train(
        int_epochs=int_epochs,
        str_mode=str_mode,
        str_metric=str_metric,
        tp_structures=tp_structures,
        dt_loaders=dt_loaders,
        model=model,
        optimiser=optimiser,
        loss=loss,
        device=device,
    )

    # save
    dt_parameters: Dict[str] = args
    int_model_number: int = model_find_number(csv_model_map=csv_model_map)
    ls_model_best, ls_history, ls_int_epoch_best = tp_model_history
    model_save_history(ls_history=ls_history, int_model_number=int_model_number, path_history=path_history)

    for int_class, str_structure in enumerate(tp_structures):
        str_model_number: str = f"{int_model_number}.{int_class}"
        dt_model: dict = ls_model_best[int_class].state_dict()
        model_save(dt_model=dt_model, str_model_number=str_model_number, path_models=path_models)

        dt_best_results_all: Dict[str, ndarray] = ls_history[ls_int_epoch_best[int_class]]
        dt_best_results: Dict[str, float] = {Terms.Headings.int_epoch: ls_int_epoch_best[int_class]}
        for str_split in Terms.tp_split:
            for int_a, metric in enumerate(Terms.Metrics.tp_metrics):
                dt_best_results[f"{str_split}_{metric}"] = dt_best_results_all[f"{str_split}"].T[int_a][int_class]
        dt_parameters.update(dt_best_results)
        dt_parameters["str_structure"] = str_structure
        dt_parameters["str_model_number"] = str_model_number
        model_save_parameters(dt_parameters=dt_parameters, csv_model_map=csv_model_map)
        Log.console(f"The best {str_structure} {str_metric} = {dt_best_results[f'{Terms.val}_{str_metric}']}.")


def model_train(
        int_epochs: int,
        str_mode: str,
        dt_loaders: Dict[str, DataLoader],
        tp_structures: Tuple[str],
        model,
        optimiser,
        loss,
        str_metric: str,
        tp_metrics=Terms.Metrics.tp_metrics,
        device=device_initialise(),
) -> Tuple:
    """train the {model} and find the optimal model based on {str_metric}"""
    int_metric = 0
    for int_metric in range(len(tp_metrics)):
        if tp_metrics[int_metric] == str_metric:
            break

    ls_history: List[Dict[str, float]] = []
    int_classes: int = len(tp_structures)
    ls_model_best = [model] * int_classes
    ls_flt_metric_best: List[float] = [0] * int_classes
    ls_int_epoch_best: List[int] = [0] * int_classes
    for int_epoch in range(int_epochs):
        Log.console("-" * 20)
        Log.console(f"Epoch: {int_epoch}")
        model, dt_metrics = model_epoch(
            str_mode=str_mode,
            dt_loaders=dt_loaders,
            model=model,
            optimiser=optimiser,
            loss=loss,
            tp_structures=tp_structures,
            device=device,
        )
        Log.console(str(dt_metrics))

        for int_class, str_structure in enumerate(tp_structures):
            flt_metric: float = dt_metrics[Terms.val][int_class].T[int_metric]
            if flt_metric > ls_flt_metric_best[int_class]:
                ls_flt_metric_best[int_class] = flt_metric
                ls_model_best[int_class] = model
                ls_int_epoch_best[int_class] = int_epoch

        dt_history: Dict = {Terms.Headings.int_epoch: int_epoch}
        dt_history.update(dt_metrics)
        ls_history.append(dt_history)

    tp_model_history: Tuple = (ls_model_best, ls_history, ls_int_epoch_best)
    return tp_model_history


def model_epoch(
        str_mode: str,
        dt_loaders: Dict[str, DataLoader],
        model,
        optimiser,
        loss,
        tp_structures: Tuple[str],
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
        ay_iou_total: ndarray = np.zeros([len(tp_structures), len(tp_metrics)])
        loader = dt_loaders[str_split]
        loop = tqdm(loader)
        ts_loss_total = 0
        int_number: int = 0
        if len(loader) - int_number != 1:
            for imgs, msks in loop:
                with torch.set_grad_enabled(str_split == Terms.train):
                    ts_imgs: Tensor = imgs.float().to(device=device)
                    ts_msks: Tensor = msks.float().to(device=device)
                    ts_prds: Tensor = model(ts_imgs).to(device=device)
                    ts_loss: Tensor = loss(ts_pred=ts_prds, ts_trth=ts_msks).to(device=device)

                    if str_split == Terms.train:
                        optimiser.zero_grad()
                        ts_loss.backward()
                        optimiser.step()

                # update loop
                ts_loss_total += ts_loss
                loop.set_postfix(loss=ts_loss_total.item())

                # update metrics
                ay_iou_total += find_batch_iou(
                    ts_msks=ts_msks,
                    ts_prds=(ts_prds > 0.5).float(),
                    tp_structures=tp_structures,
                    tp_structures_msk=tp_structures,
                    tp_metrics=tp_metrics,
                    str_mode=str_mode,
                    str_type="mask",
                )
                int_number += len(imgs)
            ay_iou_total = ay_iou_total / len(loader) * 100
            dt_metrics[str_split] = ay_iou_total
    return model, dt_metrics


if __name__ == "__main__":
    main()
