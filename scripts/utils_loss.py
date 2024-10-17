# global imports
import torch
from segmentation_models_pytorch.losses._functional import focal_loss_with_logits
from segmentation_models_pytorch.losses._functional import soft_jaccard_score
from torch import sigmoid
from torch.nn.modules.loss import _Loss

# strong typing
from torch import Tensor
from utils_main import Terms


def main():
    pass


class Loss(_Loss):
    def __init__(self, str_mode: str, str_loss: str, str_type: str = "multi"):
        self.str_mode: str = str_mode
        self.str_loss: str = str_loss
        self.str_type: str = str_type
        super().__init__()

    def forward(self, ts_pred: Tensor, ts_trth: Tensor) -> float:
        """loss forward pass"""
        int_classes: int = ts_pred.shape[1]
        int_batch: int = ts_pred.shape[0]
        loss: float = 0

        if self.str_mode in (Terms.Models.Losses.str_binary, Terms.Models.Losses.str_multilabel):
            ts_prd_class: Tensor = ts_pred.view(int_batch, int_classes, -1)
            ts_msk_class: Tensor = ts_trth.view(int_batch, int_classes, -1)
            loss += LossFunctions(ts_pred=ts_prd_class, ts_trth=ts_msk_class).run(str_loss=self.str_loss)

        if self.str_mode == Terms.Models.Losses.str_multiclass:
            for int_class in range(int_classes):
                ts_prd_class: Tensor = ts_pred[:, int_class, ...].float()
                if self.str_type == "multi":
                    ts_msk_class: Tensor = ts_trth[..., int_class].float()
                else:
                    ts_msk_class: Tensor = (ts_trth == int_class).long()
                loss += LossFunctions(ts_pred=ts_prd_class, ts_trth=ts_msk_class).run(str_loss=self.str_loss)

        if self.str_mode == "marks":
            loss += LossFunctions(ts_pred=ts_pred, ts_trth=ts_trth).run(str_loss=self.str_loss)
        return loss


class LossFunctions:
    def __init__(self, ts_pred: Tensor, ts_trth: Tensor):
        self.ts_pred: Tensor = ts_pred
        self.ts_trth: Tensor = ts_trth
        super().__init__()

    def run(self, str_loss) -> Tensor:
        """calculate loss"""
        ts_loss: Tensor = Tensor(0)
        if str_loss == Terms.Models.Losses.focal:
            ts_loss: Tensor = focal_loss_with_logits(self.ts_pred, self.ts_trth)
        if str_loss == Terms.Models.Losses.jaccard:
            ts_pred: Tensor = sigmoid(self.ts_pred)
            ts_loss: Tensor = 1 - soft_jaccard_score(ts_pred, self.ts_trth)
        if str_loss == Terms.Models.Losses.mse:
            ts_loss: Tensor = mean_squared_error(self.ts_pred, self.ts_trth)
        return ts_loss


def mean_squared_error(ts_pred: Tensor, ts_truth: Tensor) -> Tensor:
    """manual mean squared error calculation"""
    ts_loss: Tensor = torch.sum((ts_pred - ts_truth) ** 2) / len(ts_pred)
    return ts_loss


if __name__ == "__main__":
    main()
