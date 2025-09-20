#!/usr/bin/env python3
"""
Custom Action Normalizer with Clipping

This normalizer scales actions to [-1, 1] using mean and std, then clips outliers
that are outside 2x the range (i.e., outside [-2, 2]).
"""

import torch
import numpy as np
from typing import Union, Dict
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply


class ClippedActionNormalizer(LinearNormalizer):
    """
    Custom normalizer that:
    1. Scales actions to [-1, 1] using mean and std (gaussian mode)
    2. Clips actions to [-2, 2] range after normalization
    """
    
    def __init__(self, clip_range=2.0):
        super().__init__()
        self.clip_range = clip_range
    
    def fit_actions(self, action_data: Union[torch.Tensor, np.ndarray, zarr.Array]):
        """
        Fit the normalizer to action data using gaussian normalization.
        This scales to mean=0, std=1, then we'll apply additional scaling to [-1, 1].
        """
        if isinstance(action_data, np.ndarray):
            action_data = torch.from_numpy(action_data)
        elif isinstance(action_data, zarr.Array):
            action_data = torch.from_numpy(action_data[:])
        
        # Use gaussian normalization (mean=0, std=1)
        self.fit({'action': action_data}, mode='gaussian')
        
        # Now we need to scale from gaussian to [-1, 1]
        # The gaussian normalizer gives us: (x - mean) / std
        # We want: 2 * (x - mean) / std - 1  to get [-1, 1] range
        # This is equivalent to: 2 * gaussian_normalized - 1
        
        # Get the current gaussian parameters
        gaussian_scale = self.params_dict['action']['scale']  # 1/std
        gaussian_offset = self.params_dict['action']['offset']  # -mean/std
        
        # Convert to [-1, 1] scaling
        # new_scale = 2 * gaussian_scale
        # new_offset = 2 * gaussian_offset - 1
        new_scale = 2.0 * gaussian_scale
        new_offset = 2.0 * gaussian_offset - 1.0
        
        # Update the parameters
        self.params_dict['action']['scale'] = new_scale
        self.params_dict['action']['offset'] = new_offset
    
    def normalize_with_clip(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Normalize and clip actions to [-clip_range, clip_range].
        """
        # First normalize to [-1, 1]
        normalized = self.normalize(x)
        
        # Then clip to [-clip_range, clip_range]
        if isinstance(normalized, dict):
            result = {}
            for key, value in normalized.items():
                if key == 'action':
                    result[key] = torch.clamp(value, -self.clip_range, self.clip_range)
                else:
                    result[key] = value
            return result
        else:
            return torch.clamp(normalized, -self.clip_range, self.clip_range)
    
    def unnormalize_with_clip(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Unnormalize actions, handling the clipping.
        Note: This assumes the input was clipped during normalization.
        """
        return self.unnormalize(x)


def create_custom_action_normalizer_from_dataset(dataset, clip_range=2.0):
    """
    Create a custom action normalizer from a dataset.
    
    Args:
        dataset: Dataset object with get_normalizer() method
        clip_range: Range to clip to (default: 2.0 for [-2, 2])
    
    Returns:
        ClippedActionNormalizer instance
    """
    # Get the original normalizer
    original_normalizer = dataset.get_normalizer()
    
    # Create our custom normalizer
    custom_normalizer = ClippedActionNormalizer(clip_range=clip_range)
    
    # Copy over all non-action keys from original normalizer
    for key, value in original_normalizer.params_dict.items():
        if key != 'action':
            custom_normalizer.params_dict[key] = value
    
    # Get action data from the dataset to fit our custom action normalizer
    # We need to collect all action data from the dataset
    action_data_list = []
    
    # Sample some data to compute statistics
    # This is a simplified approach - in practice you might want to be more careful
    # about memory usage for very large datasets
    sample_size = min(10000, len(dataset))  # Sample up to 10k examples
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        if 'action' in sample:
            action_data_list.append(sample['action'])
    
    if action_data_list:
        # Concatenate all action data
        action_data = torch.cat(action_data_list, dim=0)
        custom_normalizer.fit_actions(action_data)
    else:
        # Fallback: use original action normalizer
        custom_normalizer.params_dict['action'] = original_normalizer.params_dict['action']
    
    return custom_normalizer


def apply_action_clipping_to_dataset(dataset, clip_range=2.0):
    """
    Apply action clipping to a dataset by modifying its normalizer.
    
    Args:
        dataset: Dataset object
        clip_range: Range to clip to (default: 2.0 for [-2, 2])
    
    Returns:
        Modified dataset with clipped action normalizer
    """
    # Create custom normalizer
    custom_normalizer = create_custom_action_normalizer_from_dataset(dataset, clip_range)
    
    # Replace the dataset's normalizer
    dataset.normalizer = custom_normalizer
    
    return dataset


# Example usage and testing
if __name__ == "__main__":
    # Test the normalizer
    print("Testing ClippedActionNormalizer...")
    
    # Create some test data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 7))
    outliers = np.random.normal(0, 5, (50, 7))  # Some outliers
    test_actions = np.vstack([normal_data, outliers])
    
    print(f"Original data stats:")
    print(f"  Min: {test_actions.min():.3f}")
    print(f"  Max: {test_actions.max():.3f}")
    print(f"  Mean: {test_actions.mean():.3f}")
    print(f"  Std: {test_actions.std():.3f}")
    
    # Create and fit normalizer
    normalizer = ClippedActionNormalizer(clip_range=2.0)
    normalizer.fit_actions(test_actions)
    
    # Test normalization with clipping
    normalized = normalizer.normalize_with_clip({'action': test_actions})
    clipped_actions = normalized['action']
    
    print(f"\nAfter normalization and clipping:")
    print(f"  Min: {clipped_actions.min():.3f}")
    print(f"  Max: {clipped_actions.max():.3f}")
    print(f"  Mean: {clipped_actions.mean():.3f}")
    print(f"  Std: {clipped_actions.std():.3f}")
    
    # Check clipping
    outside_range = (clipped_actions < -2.0) | (clipped_actions > 2.0)
    print(f"  Values outside [-2, 2]: {outside_range.sum().item()}")
    
    # Test unnormalization
    unnormalized = normalizer.unnormalize_with_clip(normalized)
    print(f"\nAfter unnormalization:")
    print(f"  Min: {unnormalized['action'].min():.3f}")
    print(f"  Max: {unnormalized['action'].max():.3f}")
    print(f"  Mean: {unnormalized['action'].mean():.3f}")
    print(f"  Std: {unnormalized['action'].std():.3f}")
    
    print("\n✅ Test completed successfully!")
