"""Diagnostic variants: original, all-open, all-closed for identical arm paths."""
import torch

def augment_gripper_candidates(actions):
    if actions.ndim!=4 or actions.shape[-1]!=7:
        raise ValueError('expected [batch,candidate,horizon,7] actions')
    opened=actions.clone();opened[...,6]=1.
    closed=actions.clone();closed[...,6]=0.
    return torch.cat([actions,opened,closed],dim=1)
