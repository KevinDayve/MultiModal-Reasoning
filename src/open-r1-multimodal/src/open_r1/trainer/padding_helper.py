"""
Helper functions to ensure left side padding consistency across generations.
"""
import torch

def enforceLeftPad(
        input_ids,
        pad_token_id
):
    """
    Ensures that the input_ids are left-padded to the same length as the longest sequence in the batch.
    Args:
        input_ids: List of input IDs (tokenized sequences).
        pad_token_id: The token ID used for padding.
    Returns:
        Padded input IDs.
    """
    max_length = input_ids.shape[1]
    fixed_ids = []
    for row in input_ids:
        tokens = row[row != pad_token_id]
        num_pad = max_length - tokens.size(0)
        padded = torch.cat([torch.full((num_pad,), pad_token_id, dtype=torch.long, device=tokens.device), tokens])
        fixed_ids.append(padded)
    return torch.stack(fixed_ids)

def enforceLeftAttention(
        input_ids,
        pad_token_id,
):
    return (input_ids != pad_token_id).long()