from typing import List, Tuple, Union
import torch
from torch import Tensor, tensor
def _levenshtein_distance(prediction_tokens: List[str], reference_tokens: List[str]) -> int:
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(prediction_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]

def _cer_update(
    preds: Union[str, List[str]],
    target: Union[str, List[str]],
) -> Tuple[Tensor, Tensor]:

    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    errors = tensor(0, dtype=torch.float)
    total = tensor(0, dtype=torch.float)
    for pred, tgt in zip(preds, target):
        pred_tokens = pred
        tgt_tokens = tgt
        errors += _levenshtein_distance(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)
    return errors, total

def char_error_rate(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    errors, total = _cer_update(preds, target)
    return errors / total