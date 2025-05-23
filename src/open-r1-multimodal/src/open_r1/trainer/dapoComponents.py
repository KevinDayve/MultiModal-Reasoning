import torch
from trl.core import masked_mean
def DAPOAdvantage(rewards, num_generations, coefficient=1e-4):
    """
    Function which computes the grouped rewards, and then returns the advantage for DAPO.
    Arguments:
        rewards: (torch.Tensor): Tensor of rewards
        num_generations: int: the "G" in Grpo, or number of generations per prompt.
        coefficient: Small constant to avoid zero division error.
    Returns:
        torch.Tensor: The advantage for DAPO (The advantage function is the same as GRPO advantage function.)
    """
    rewards = rewards.view(-1, num_generations)
    meanRewards = torch.mean(rewards, dim=1, keepdim=True)
    stdRewards = torch.std(rewards, dim=1, keepdim=True)
    advantage = (rewards - meanRewards) / (stdRewards + coefficient)
    return advantage.view(-1)

def DAPOLoss(
        advantage,
        log_probs,
        log_probs_ref,
        mask,
        epsilon_low=0.2,
        epsilon_high=0.28,
        token_level=True):
    """
    Objective function for DAPO.
    Arguments:
        advantages: Tensor containing advantages.
        log_probs: logits generated by the under RL model.
        log_probs_ref: logits generated by the reference (base) model.
        mask: The completion mask
        epsilon_low: Lower clipping bound.
        epsilon_high: Upper clipping bound.
        token_level: Whether to compute the loss at a token level (Defaults to True). If set to False, it will compute the loss at a sequence level.
    Returns:
        torch.Tensor: The loss for DAPO.
    """
    advantages = advantage.unsqueeze(1).expand_as(log_probs)
    #We will compute the ratio.
    ratios = torch.exp(log_probs-log_probs_ref)

    #PPO-esque clipping objective.
    clippedRatios = torch.clamp(ratios, min=epsilon_low, max=epsilon_high)
    #This helps in preserving the gradients
    lossPerToken = torch.exp(log_probs - log_probs.detach()) * advantages
    lossPerToken = -torch.min(advantages * ratios, advantages * clippedRatios)

    #Now we will do token-level averaging
    if token_level: #True for DAPO.
        loss = masked_mean(
            lossPerToken,
            mask
        )
    else: #Same as GRPO aka not token level.
        loss = ((lossPerToken * mask).sum(dim=1) / mask.sum(dim=1)).mean()
    return loss