#!/usr/bin/env python3
"""
Data Distribution Analysis Script - Low-Dim Focus with Nested Structure Support

This script analyzes the distribution of low-dimensional data (obs, actions) in a given directory,
providing comprehensive statistics about data ranges, outliers, and distribution characteristics.
Automatically skips image data to avoid memory issues. Handles nested zarr structures.

Usage:
    python analyze_data_distribution.py --data_dir /path/to/data --file_pattern "*.zarr"
    python analyze_data_distribution.py --data_dir /path/to/data --file_pattern "*.hdf5" --max_files 10
    python analyze_data_distribution.py --data_dir /path/to/data --file_pattern "*.zarr" --keys obs action --outlier_method iqr
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class DataAnalyzer:
    def __init__(self, outlier_method='iqr', outlier_threshold=1.5, dimension_breakdown=True):
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.dimension_breakdown = dimension_breakdown
        self.stats = {}
        self.file_stats = []
        
    def is_image_data(self, key: str, shape: tuple) -> bool:
        """Check if data appears to be image data based on key name and shape."""
        # Common image key patterns
        image_key_patterns = [
            'image', 'img', 'rgb', 'depth', 'camera', 'visual', 'pixel',
            'obs_image', 'obs_rgb', 'obs_depth', 'obs_camera', 'obs_visual',
            'front_rgb', 'side_rgb', 'wrist_rgb', 'top_rgb', 'bottom_rgb'
        ]
        
        # Check key name
        key_lower = key.lower()
        for pattern in image_key_patterns:
            if pattern in key_lower:
                return True
        
        # Check shape - images typically have 3+ dimensions with height/width
        if len(shape) >= 3:
            # If last 2 dimensions look like height/width (reasonable image sizes)
            if shape[-1] > 32 and shape[-2] > 32:
                return True
        
        return False
    
    def is_lowdim_data(self, key: str, shape: tuple) -> bool:
        """Check if data appears to be low-dimensional (obs, actions, etc.)."""
        # Common low-dim key patterns
        lowdim_key_patterns = [
            'obs', 'action', 'reward', 'done', 'info', 'state', 'pose',
            'position', 'velocity', 'joint', 'gripper', 'force', 'torque',
            'arm_joint_pos', 'end_effector_pose', 'fixed_asset_pose',
            'held_asset_pose', 'last_arm_action', 'last_gripper_action',
            'last_processed_arm_action', 'held_asset_in_fixed_asset_frame'
        ]
        
        # Check key name
        key_lower = key.lower()
        for pattern in lowdim_key_patterns:
            if pattern in key_lower:
                return True
        
        # Check shape - low-dim data typically has 1-2 dimensions
        if len(shape) <= 2:
            return True
        
        return False
    
    def should_analyze_key(self, key: str, shape: tuple) -> bool:
        """Determine if we should analyze this key (skip images, focus on low-dim)."""
        # Skip image data
        if self.is_image_data(key, shape):
            return False
        
        # Focus on low-dimensional data
        return self.is_lowdim_data(key, shape)
    
    def detect_outliers(self, data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
        """Detect outliers using specified method."""
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            if SCIPY_AVAILABLE:
                z_scores = np.abs(stats.zscore(data))
                return z_scores > threshold
            else:
                # Fallback to manual z-score calculation
                z_scores = np.abs((data - np.mean(data)) / np.std(data))
                return z_scores > threshold
        
        elif method == 'modified_zscore':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            return np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def analyze_array(self, data: np.ndarray, key: str, file_path: str) -> Dict:
        """Analyze a single array and return statistics."""
        if data.size == 0:
            return None
        
        # Basic statistics
        stats = {
            'key': key,
            'file': file_path,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'size': data.size,
            'memory_mb': data.nbytes / (1024 * 1024),
        }
        
        # Analyze each dimension if it's multi-dimensional and dimension_breakdown is enabled
        if self.dimension_breakdown and len(data.shape) > 1 and data.shape[-1] <= 50:  # Only for reasonable dimension sizes
            stats['dimension_analysis'] = self.analyze_dimensions(data, key)
        
        # Flatten for overall analysis
        flat_data = data.flatten()
        
        # Overall statistics
        stats.update({
            'mean': np.mean(flat_data),
            'std': np.std(flat_data),
            'min': np.min(flat_data),
            'max': np.max(flat_data),
            'range': np.max(flat_data) - np.min(flat_data),
            'q25': np.percentile(flat_data, 25),
            'q50': np.percentile(flat_data, 50),
            'q75': np.percentile(flat_data, 75),
            'iqr': np.percentile(flat_data, 75) - np.percentile(flat_data, 25),
        })
        
        # Advanced statistics if scipy is available
        if SCIPY_AVAILABLE:
            try:
                stats['skewness'] = stats.skew(flat_data)
                stats['kurtosis'] = stats.kurtosis(flat_data)
            except:
                stats['skewness'] = None
                stats['kurtosis'] = None
        
        # Outlier detection
        try:
            outliers = self.detect_outliers(flat_data, self.outlier_method, self.outlier_threshold)
            stats['outlier_count'] = np.sum(outliers)
            stats['outlier_percentage'] = (np.sum(outliers) / len(flat_data)) * 100
        except:
            stats['outlier_count'] = 0
            stats['outlier_percentage'] = 0.0
        
        # Distribution type detection
        stats['distribution_type'] = self.classify_distribution(flat_data)
        
        return stats
    
    def analyze_dimensions(self, data: np.ndarray, key: str) -> Dict:
        """Analyze each dimension of multi-dimensional data."""
        if len(data.shape) < 2:
            return {}
        
        # Reshape to (samples, features) for analysis
        if len(data.shape) == 2:
            reshaped_data = data
        else:
            # Flatten all dimensions except the last one
            reshaped_data = data.reshape(-1, data.shape[-1])
        
        n_dims = reshaped_data.shape[1]
        dim_stats = {}
        
        for dim in range(n_dims):
            dim_data = reshaped_data[:, dim]
            dim_key = f"dim_{dim}"
            
            # Basic statistics for this dimension
            dim_stats[dim_key] = {
                'mean': np.mean(dim_data),
                'std': np.std(dim_data),
                'min': np.min(dim_data),
                'max': np.max(dim_data),
                'range': np.max(dim_data) - np.min(dim_data),
                'q25': np.percentile(dim_data, 25),
                'q50': np.percentile(dim_data, 50),
                'q75': np.percentile(dim_data, 75),
                'iqr': np.percentile(dim_data, 75) - np.percentile(dim_data, 25),
            }
            
            # Outlier detection for this dimension
            try:
                outliers = self.detect_outliers(dim_data, self.outlier_method, self.outlier_threshold)
                dim_stats[dim_key]['outlier_count'] = np.sum(outliers)
                dim_stats[dim_key]['outlier_percentage'] = (np.sum(outliers) / len(dim_data)) * 100
            except:
                dim_stats[dim_key]['outlier_count'] = 0
                dim_stats[dim_key]['outlier_percentage'] = 0.0
            
            # Distribution type for this dimension
            dim_stats[dim_key]['distribution_type'] = self.classify_distribution(dim_data)
            
            # Advanced statistics if scipy is available
            if SCIPY_AVAILABLE:
                try:
                    dim_stats[dim_key]['skewness'] = stats.skew(dim_data)
                    dim_stats[dim_key]['kurtosis'] = stats.kurtosis(dim_data)
                except:
                    dim_stats[dim_key]['skewness'] = None
                    dim_stats[dim_key]['kurtosis'] = None
        
        return dim_stats
    
    def classify_distribution(self, data: np.ndarray) -> str:
        """Classify the distribution type of the data."""
        if len(data) < 10:
            return 'insufficient_data'
        
        # Check for constant data
        if np.std(data) < 1e-10:
            return 'constant'
        
        # Check for uniform distribution
        if SCIPY_AVAILABLE:
            try:
                _, p_value = stats.kstest(data, 'uniform')
                if p_value > 0.05:
                    return 'uniform'
            except:
                pass
        
        # Check for normal distribution
        if SCIPY_AVAILABLE:
            try:
                _, p_value = stats.normaltest(data)
                if p_value > 0.05:
                    return 'normal'
            except:
                pass
        
        # Check for uniform-like distribution (manual check)
        q25, q75 = np.percentile(data, [25, 75])
        if q75 - q25 > 0.8 * (np.max(data) - np.min(data)):
            return 'uniform_like'
        
        return 'unknown'
    
    def load_zarr_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load data from a Zarr file, handling nested structures."""
        if not ZARR_AVAILABLE:
            raise ImportError("Zarr not available. Install with: pip install zarr")
        
        data = {}
        try:
            store = zarr.open(file_path, mode='r')
            
            def load_recursive(group, prefix=''):
                """Recursively load data from nested zarr groups."""
                for key, value in group.items():
                    full_key = f"{prefix}/{key}" if prefix else key
                    
                    if hasattr(value, 'shape'):
                        # This is a dataset/array
                        if self.should_analyze_key(full_key, value.shape):
                            print(f"  🔍 Loading low-dim data: {full_key} (shape: {value.shape})")
                            data[full_key] = value[:]
                        else:
                            print(f"  ⏭️  Skipping image data: {full_key} (shape: {value.shape})")
                    elif hasattr(value, 'keys'):
                        # This is a group, recurse into it
                        load_recursive(value, full_key)
            
            load_recursive(store)
            
        except Exception as e:
            print(f"  ❌ Error loading Zarr file: {e}")
            return {}
        
        return data
    
    def load_hdf5_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load data from an HDF5 file."""
        if not H5PY_AVAILABLE:
            raise ImportError("H5py not available. Install with: pip install h5py")
        
        data = {}
        try:
            with h5py.File(file_path, 'r') as f:
                def load_recursive(group, prefix=''):
                    for key, value in group.items():
                        full_key = f"{prefix}/{key}" if prefix else key
                        if isinstance(value, h5py.Dataset):
                            # Check if we should analyze this key
                            if self.should_analyze_key(full_key, value.shape):
                                data[full_key] = value[:]
                            else:
                                print(f"  ⏭️  Skipping image data: {full_key} (shape: {value.shape})")
                        elif isinstance(value, h5py.Group):
                            load_recursive(value, full_key)
                
                load_recursive(f)
        except Exception as e:
            print(f"  ❌ Error loading HDF5 file: {e}")
            return {}
        
        return data
    
    def load_npz_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load data from an NPZ file."""
        data = {}
        try:
            with np.load(file_path) as f:
                for key in f.keys():
                    # Check if we should analyze this key
                    if self.should_analyze_key(key, f[key].shape):
                        data[key] = f[key]
                    else:
                        print(f"  ⏭️  Skipping image data: {key} (shape: {f[key].shape})")
        except Exception as e:
            print(f"  ❌ Error loading NPZ file: {e}")
            return {}
        
        return data
    
    def load_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load data from a file based on its extension."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.zarr':
            return self.load_zarr_file(str(file_path))
        elif file_path.suffix in ['.h5', '.hdf5']:
            return self.load_hdf5_file(str(file_path))
        elif file_path.suffix == '.npz':
            return self.load_npz_file(str(file_path))
        else:
            print(f"  ❌ Unsupported file format: {file_path.suffix}")
            return {}
    
    def analyze_file(self, file_path: str) -> List[Dict]:
        """Analyze a single file and return statistics for all keys."""
        print(f"📁 Analyzing: {file_path}")
        
        data = self.load_file(file_path)
        if not data:
            return []
        
        file_stats = []
        for key, array in data.items():
            print(f"  🔍 Analyzing key: {key} (shape: {array.shape})")
            stats = self.analyze_array(array, key, file_path)
            if stats:
                file_stats.append(stats)
        
        return file_stats
    
    def analyze_directory(self, data_dir: str, file_pattern: str = "*.zarr", 
                         max_files: Optional[int] = None, recursive: bool = False) -> None:
        """Analyze all files in a directory matching the pattern."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"❌ Directory does not exist: {data_dir}")
            return
        
        # Find files
        if recursive:
            pattern = f"**/{file_pattern}"
        else:
            pattern = file_pattern
        
        files = list(data_dir.glob(pattern))
        
        if not files:
            print(f"❌ No files found matching pattern: {file_pattern}")
            return
        
        # Limit files if specified
        if max_files:
            files = files[:max_files]
        
        print(f"🔍 Found {len(files)} files to analyze")
        print(f"📊 Focus: Low-dimensional data only (obs, actions, etc.)")
        print(f"⏭️  Skipping: Image data to avoid memory issues")
        print(f"🔬 Dimension breakdown: {'Enabled' if self.dimension_breakdown else 'Disabled'}")
        print("=" * 80)
        
        # Analyze files
        all_stats = []
        successful_files = 0
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
            try:
                file_stats = self.analyze_file(str(file_path))
                all_stats.extend(file_stats)
                successful_files += 1
                print(f"  ✅ Successfully analyzed {len(file_stats)} keys")
            except Exception as e:
                print(f"  ❌ Error analyzing file: {e}")
        
        # Aggregate statistics
        self.aggregate_stats(all_stats, successful_files, len(files))
    
    def aggregate_stats(self, all_stats: List[Dict], successful_files: int, total_files: int) -> None:
        """Aggregate statistics across all files and keys."""
        if not all_stats:
            print("\n❌ No data to analyze")
            return
        
        print("\n" + "=" * 80)
        print("📊 DATA DISTRIBUTION ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total files processed: {total_files}")
        print(f"  Successful: {successful_files}")
        print(f"  Failed: {total_files - successful_files}")
        
        # Data overview
        total_size = sum(stat['memory_mb'] for stat in all_stats)
        unique_keys = list(set(stat['key'] for stat in all_stats))
        
        print(f"\n📁 DATA OVERVIEW:")
        print(f"  Total data size: {total_size:.2f} MB")
        print(f"  Unique data keys: {len(unique_keys)}")
        print(f"  Keys found: {sorted(unique_keys)}")
        
        # Per-key analysis
        print(f"\n🔍 DETAILED ANALYSIS BY KEY:")
        print("-" * 80)
        
        for key in sorted(unique_keys):
            key_stats = [stat for stat in all_stats if stat['key'] == key]
            
            print(f"\n📋 Key: '{key}'")
            print("-" * 40)
            
            # Aggregate statistics for this key
            total_samples = sum(stat['size'] for stat in key_stats)
            files_with_key = len(key_stats)
            
            # Global statistics
            all_values = []
            for stat in key_stats:
                # Sample some values for global stats (to avoid memory issues)
                if stat['size'] > 1000000:  # If very large, sample
                    sample_size = min(100000, stat['size'])
                    indices = np.random.choice(stat['size'], sample_size, replace=False)
                    # This is a simplified approach - in practice you'd need to reconstruct the data
                    all_values.extend([stat['mean']] * sample_size)  # Simplified
                else:
                    all_values.extend([stat['mean']] * stat['size'])  # Simplified
            
            if all_values:
                global_mean = np.mean(all_values)
                global_std = np.std(all_values)
                global_min = min(stat['min'] for stat in key_stats)
                global_max = max(stat['max'] for stat in key_stats)
            else:
                global_mean = global_std = global_min = global_max = 0
            
            print(f"  Total samples: {total_samples:,}")
            print(f"  Files containing this key: {files_with_key}")
            print(f"  Mean across files: {global_mean:.6f}")
            print(f"  Std across files: {global_std:.6f}")
            print(f"  Global min: {global_min:.6f}")
            print(f"  Global max: {global_max:.6f}")
            print(f"  Global range: {global_max - global_min:.6f}")
            
            # Outlier analysis
            total_outliers = sum(stat['outlier_count'] for stat in key_stats)
            total_outlier_percentage = (total_outliers / total_samples) * 100 if total_samples > 0 else 0
            
            print(f"  Total outliers: {total_outliers:,} ({total_outlier_percentage:.2f}%)")
            
            # Data types and shapes
            dtypes = list(set(stat['dtype'] for stat in key_stats))
            shapes = list(set(str(stat['shape']) for stat in key_stats))
            distribution_types = list(set(stat['distribution_type'] for stat in key_stats))
            
            print(f"  Data types: {dtypes}")
            print(f"  Shapes: {shapes}")
            print(f"  Distribution types: {distribution_types}")
            
            # Memory usage
            total_memory = sum(stat['memory_mb'] for stat in key_stats)
            print(f"  Total memory usage: {total_memory:.2f} MB")
            
            # Dimension breakdown if available
            if self.dimension_breakdown and any('dimension_analysis' in stat for stat in key_stats):
                print(f"\n  🔬 DIMENSION BREAKDOWN:")
                self.print_dimension_breakdown(key_stats)
            
            # Outlier breakdown by file
            print(f"\n  🚨 OUTLIER ANALYSIS BY FILE:")
            for stat in key_stats:
                file_name = Path(stat['file']).name
                outlier_pct = (stat['outlier_count'] / stat['size']) * 100 if stat['size'] > 0 else 0
                print(f"    {file_name}: {stat['outlier_count']:,} outliers ({outlier_pct:.2f}%)")
        
        print("\n" + "=" * 80)
        print("✅ Analysis complete!")
    
    def print_dimension_breakdown(self, key_stats: List[Dict]) -> None:
        """Print detailed breakdown by dimension for multi-dimensional data."""
        # Collect all dimension data
        all_dim_stats = {}
        
        for stat in key_stats:
            if 'dimension_analysis' in stat:
                for dim_key, dim_stat in stat['dimension_analysis'].items():
                    if dim_key not in all_dim_stats:
                        all_dim_stats[dim_key] = []
                    all_dim_stats[dim_key].append(dim_stat)
        
        if not all_dim_stats:
            print("    No dimension breakdown available")
            return
        
        # Print statistics for each dimension
        for dim_key in sorted(all_dim_stats.keys()):
            dim_num = dim_key.split('_')[1]
            dim_stats_list = all_dim_stats[dim_key]
            
            # Aggregate across files for this dimension
            mean_vals = [ds['mean'] for ds in dim_stats_list]
            std_vals = [ds['std'] for ds in dim_stats_list]
            min_vals = [ds['min'] for ds in dim_stats_list]
            max_vals = [ds['max'] for ds in dim_stats_list]
            outlier_counts = [ds['outlier_count'] for ds in dim_stats_list]
            outlier_pcts = [ds['outlier_percentage'] for ds in dim_stats_list]
            
            print(f"    📐 Dimension {dim_num}:")
            print(f"      Mean: {np.mean(mean_vals):.6f} ± {np.std(mean_vals):.6f}")
            print(f"      Std:  {np.mean(std_vals):.6f} ± {np.std(std_vals):.6f}")
            print(f"      Range: [{np.min(min_vals):.6f}, {np.max(max_vals):.6f}]")
            print(f"      Outliers: {sum(outlier_counts):,} ({np.mean(outlier_pcts):.2f}% ± {np.std(outlier_pcts):.2f}%)")
            
            # Distribution types for this dimension
            dist_types = [ds['distribution_type'] for ds in dim_stats_list]
            unique_dists = list(set(dist_types))
            print(f"      Distribution: {unique_dists}")

def main():
    parser = argparse.ArgumentParser(description='Analyze data distribution in directory')
    parser.add_argument('--data_dir', required=True, help='Directory containing data files')
    parser.add_argument('--file_pattern', default='*.zarr', help='File pattern to match (default: *.zarr)')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process')
    parser.add_argument('--outlier_method', choices=['iqr', 'zscore', 'modified_zscore'], 
                       default='iqr', help='Outlier detection method (default: iqr)')
    parser.add_argument('--outlier_threshold', type=float, default=1.5, 
                       help='Threshold for outlier detection (default: 1.5)')
    parser.add_argument('--recursive', action='store_true', 
                       help='Search recursively in subdirectories')
    parser.add_argument('--no_dimension_breakdown', action='store_true',
                       help='Disable dimension-by-dimension analysis')
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.file_pattern.endswith('.zarr') and not ZARR_AVAILABLE:
        print("❌ Zarr not available. Install with: pip install zarr")
        sys.exit(1)
    
    if args.file_pattern.endswith(('.h5', '.hdf5')) and not H5PY_AVAILABLE:
        print("❌ H5py not available. Install with: pip install h5py")
        sys.exit(1)
    
    # Create analyzer and run analysis
    analyzer = DataAnalyzer(
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold,
        dimension_breakdown=not args.no_dimension_breakdown
    )
    
    analyzer.analyze_directory(
        data_dir=args.data_dir,
        file_pattern=args.file_pattern,
        max_files=args.max_files,
        recursive=args.recursive
    )

if __name__ == "__main__":
    main()
