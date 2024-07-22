from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor, tensor

from torchmetrics.functional.regression.mse import _mean_squared_error_compute, _mean_squared_error_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from medpy import metric


if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MeanSquaredError.plot"]

"""
class MeanSquaredError(Metric):

    is_differentiable = True #求导
    higher_is_better = False # 较低的mse数值可能表现出更好的性能
    full_state_update = False # 更新状态完整数据
    plot_lower_bound: float = 0.0 # 绘图时使用最小界限

    sum_squared_error: Tensor # 存储累加的平方误差综合
    total: Tensor # 存储样本数量综合

    def __init__(
        self,
        squared: bool = True,
        num_outputs: int = 1, # 输出数量
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(squared, bool):
            raise ValueError(f"Expected argument `squared` to be a boolean but got {squared}")
        self.squared = squared

        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(f"Expected num_outputs to be a positive integer but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("sum_squared_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    # 计算当前批次的mse，并更新
    def update(self, preds: Tensor, target: Tensor) -> None:
        sum_squared_error, num_obs = _mean_squared_error_update(preds, target, num_outputs=self.num_outputs)

        self.sum_squared_error += sum_squared_error
        self.total += num_obs

    # 使用累积的平方误差和样本总数来计算整个数据集上的MSE
    def compute(self) -> Tensor:
        return _mean_squared_error_compute(self.sum_squared_error, self.total, squared=self.squared)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:

        return self._plot(val, ax)
        
"""

class DiceCoefficient(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_dice", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_volumes", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.total_dice += torch.tensor(metric.binary.dc(preds.cpu().numpy(), target.cpu().numpy()))
        self.total_volumes += 1

    def compute(self):
        return self.total_dice / self.total_volumes

class JaccardIndex(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_jaccard", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_volumes", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.total_jaccard += torch.tensor(metric.binary.jc(preds.cpu().numpy(), target.cpu().numpy()))
        self.total_volumes += 1

    def compute(self):
        return self.total_jaccard / self.total_volumes


class AverageSurfaceDistance(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_asd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_volumes", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        asd_score = torch.tensor(metric.binary.asd(preds.cpu().numpy(), target.cpu().numpy()))
        self.total_asd += asd_score
        self.total_volumes += 1

    def compute(self):
        return self.total_asd / self.total_volumes


class HausdorffDistance95(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_hd95", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_volumes", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        hd95_score = torch.tensor(metric.binary.hd95(preds.cpu().numpy(), target.cpu().numpy()))
        self.total_hd95 += hd95_score
        self.total_volumes += 1

    def compute(self):
        return self.total_hd95 / self.total_volumes
