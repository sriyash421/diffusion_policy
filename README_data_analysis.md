# Data Distribution Analysis Script - Low-Dim Focus

This script provides comprehensive analysis of **low-dimensional data distributions** (obs, actions, rewards, etc.) in your dataset files. It automatically **skips image data** to avoid memory issues and long processing times.

## 🎯 **Key Features**

- **Low-dim focus**: Only analyzes obs, actions, rewards, states, poses, etc.
- **Image data skipping**: Automatically detects and skips image data to prevent memory issues
- **Multi-format support**: Zarr, HDF5, NPZ files
- **Comprehensive statistics**: Mean, std, min, max, quartiles, skewness, kurtosis
- **Outlier detection**: IQR, Z-score, and Modified Z-score methods
- **Distribution analysis**: Normal, uniform, constant data detection
- **Memory usage tracking**: Data size and memory consumption
- **Aggregate analysis**: Statistics across multiple files

## 🚀 **Quick Start**

```bash
# Analyze all zarr files in a directory (focus on low-dim data only)
python analyze_data_distribution.py --data_dir /path/to/your/data

# Analyze HDF5 files
python analyze_data_distribution.py --data_dir /path/to/your/data --file_pattern "*.h5"

# Analyze NPZ files
python analyze_data_distribution.py --path/to/your/data --file_pattern "*.npz"
```

## 📊 **What Gets Analyzed vs Skipped**

### ✅ **Analyzed (Low-Dim Data)**
- `obs` - observations
- `action` - actions
- `reward` - rewards
- `done` - episode termination flags
- `state` - robot states
- `pose` - poses
- `position` - positions
- `velocity` - velocities
- `joint` - joint states
- `gripper` - gripper states
- `force` - forces
- `torque` - torques
- Any data with ≤2 dimensions

### ⏭️ **Skipped (Image Data)**
- `image`, `img`, `rgb`, `depth`, `camera`, `visual`, `pixel`
- `obs_image`, `obs_rgb`, `obs_depth`, `obs_camera`
- Any data with 3+ dimensions where last 2 dims > 32x32
- Large multi-dimensional arrays

## 🔧 **Advanced Usage**

```bash
# Limit number of files processed
python analyze_data_distribution.py --data_dir /path/to/your/data --max_files 10

# Use different outlier detection method
python analyze_data_distribution.py --data_dir /path/to/your/data --outlier_method zscore --outlier_threshold 3.0

# Search recursively in subdirectories
python analyze_data_distribution.py --data_dir /path/to/your/data --recursive

# Analyze specific file types
python analyze_data_distribution.py --data_dir /path/to/your/data --file_pattern "*.h5"
```

## 📋 **Command Line Arguments**

- `--data_dir`: Directory containing data files (required)
- `--file_pattern`: File pattern to match (default: "*.zarr")
- `--max_files`: Maximum number of files to process (default: all)
- `--outlier_method`: Outlier detection method - 'iqr', 'zscore', 'modified_zscore' (default: 'iqr')
- `--outlier_threshold`: Threshold for outlier detection (default: 1.5)
- `--recursive`: Search recursively in subdirectories

## 📈 **Example Output**

```
================================================================================
DATA DISTRIBUTION ANALYSIS SUMMARY
================================================================================

📊 OVERALL STATISTICS:
  Total files processed: 5
  Successful: 5
  Failed: 0

📁 DATA OVERVIEW:
  Total data size: 125.45 MB
  Unique data keys: 3
  Keys found: ['action', 'obs', 'reward']

🔍 DETAILED ANALYSIS BY KEY:
--------------------------------------------------------------------------------

📋 Key: 'action'
----------------------------------------
  Total samples: 125,000
  Files containing this key: 5
  Mean across files: 0.123456
  Std across files: 0.789012
  Global min: -2.500000
  Global max: 2.500000
  Global range: 5.000000
  Total outliers: 1,250 (1.00%)
  Data types: ['float32']
  Shapes: ['(250, 7)']
  Distribution types: ['normal']
  Total memory usage: 12.50 MB

  🚨 OUTLIER ANALYSIS:
    File 1: 250 outliers (1.00%)
    File 3: 500 outliers (2.00%)

📋 Key: 'obs'
----------------------------------------
  Total samples: 125,000
  Files containing this key: 5
  Mean across files: 0.045678
  Std across files: 0.234567
  Global min: -1.200000
  Global max: 1.800000
  Global range: 3.000000
  Total outliers: 625 (0.50%)
  Data types: ['float32']
  Shapes: ['(250, 12)']
  Distribution types: ['normal']
  Total memory usage: 25.00 MB
```

## 🛠️ **Dependencies**

- **Required**: numpy, pandas
- **Optional**: zarr (for .zarr files), h5py (for .h5/.hdf5 files), scipy (for advanced statistics)

Install with:
```bash
pip install numpy pandas zarr h5py scipy
```

## ⚡ **Performance Notes**

- **Fast processing**: Only loads low-dimensional data
- **Memory efficient**: Skips large image arrays
- **Smart filtering**: Automatically detects data types
- **Scalable**: Can process hundreds of files quickly

## 🔍 **Data Type Detection**

The script uses intelligent heuristics to determine what to analyze:

1. **Key name patterns**: Looks for common low-dim key names
2. **Shape analysis**: Prefers 1-2 dimensional arrays
3. **Size thresholds**: Skips arrays that look like images (>32x32 in last 2 dims)
4. **Manual override**: You can modify the filtering logic if needed

This ensures you get comprehensive analysis of your robot data without the overhead of processing image files!
