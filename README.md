# Hucenrotia TM5-700 Screwing PSU Dataset

This project involves collecting multimodal robotic screwing data using the TM5-700 robot and DynPick force sensor. It includes real-time force sensing, visual recording, and semantic command annotation using a GNN-based command generator.


## ✅ Prerequisites

Before running the scripts, set up the environment:

```bash
conda create -n record_h5py python=3.10
conda activate record_h5py
pip3 install -r requirements.txt
```

## 🔧 Building the DynPick Library

```bash
cd scripts/read_dynpick_force_sensor
mkdir build
cd build
cmake ..
make
```

This builds the C++ interface for reading the DynPick USB force sensor.

## 🛑 Fixing USB Permission Denied Issue

If you encounter a `Permission denied` error on `/dev/ttyUSB0`:

1. **Check port ownership:**
   ```bash
   ls -l /dev/ttyUSB0
   # Expected: crw-rw---- 1 root dialout ...
   ```

2. **Add yourself to the dialout group:**
   ```bash
   sudo usermod -aG dialout $USER
   ```

3. **Apply the new group permission:**
   ```bash
   newgrp dialout
   ```
   Or simply reboot your system.

## 🧠 Generate GNN Command Dataset

To create semantic command pairs (action–object combinations), run:

```bash
python generate_commands.py
```

This script outputs a dataset compatible with graph-based command models.

## 📹 Record Dataset with Force and Visual Data

1. Place `wacoh_sensor.cpython-310-x86_64-linux-gnu.so` in the project root (same location as `record_dataset_h5py.py`).
2. Start recording:
   ```bash
   python record_dataset_h5py.py
   ```

This will collect synchronized:
- Force data from DynPick
- Action commands
- Object annotations
- (Optional) Visual or wrist-camera inputs

## 📦 Project Structure

```
├── generate_commands.py              # Script to generate GNN command dataset
├── record_dataset_h5py.py            # Dataset recording tool
├── wacoh_sensor.cpython-310-x86_64-linux-gnu.so  # Compiled sensor interface
├── scripts/
    ├── read_dynpick_force_sensor/        # DynPick driver

```

## 📁 Output

The recorded dataset is saved in HDF5 format for training multimodal learning models such as:
- Visual language action policies
- Force aware policy learning

## ✍️ Author

Developed by Nurdin Khoirurizka from Hucenrotia Lab — Generative Artificial Inteligent Research Group
