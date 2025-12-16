# Stable Derolling: First Stage Derolling Pipeline using Marigold Framework


## 🛠️ Setup

The inference code was tested on:

- Ubuntu 22.04 LTS, Python 3.10.12,  CUDA 11.7, GeForce RTX 3090 (pip, Mamba)
- CentOS Linux 7, Python 3.10.4, CUDA 11.7, GeForce RTX 4090 (pip)
- Windows 11 22H2, Python 3.10.12, CUDA 12.3, GeForce RTX 3080 (Mamba)
- MacOS 14.2, Python 3.10.12, M1 16G (pip)

### 🪧 A Note for Windows users

We recommend running the code in WSL2:

1. Install WSL following [installation guide](https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command).
1. Install CUDA support for WSL following [installation guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2).
1. Find your drives in `/mnt/<drive letter>/`; check [WSL FAQ](https://learn.microsoft.com/en-us/windows/wsl/faq#how-do-i-access-my-c--drive-) for more details. Navigate to the working directory of choice. 

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/JacobACarter/Stable-Derolling.git
cd Marigold
```

### 💻 Dependencies

We provide several ways to install the dependencies.

1. **Using [Mamba](https://github.com/mamba-org/mamba)**, which can installed together with [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3). 

    Windows users: Install the Linux version into the WSL.

    After the installation, Miniforge needs to be activated first: `source /home/$USER/miniforge3/bin/activate`.

    Create the environment and install dependencies into it:

    ```bash
    mamba env create -n marigold --file environment.yaml
    conda activate marigold
    ```

2. **Using pip:** 
    Alternatively, create a Python native virtual environment and install dependencies into it:

    ```bash
    python -m venv venv/marigold
    source venv/marigold/bin/activate
    pip install -r requirements.txt
    ```

Keep the environment activated before running the inference script. 
Activate the environment again after restarting the terminal session.



At inference, specify the checkpoint path: Use the test_run.py file!

```bash
python test_run.py \
    --checkpoint checkpoint/marigold-v1-0 \
    --denoise_steps 50 \
    --ensemble_size 1 \
    --input_rgb_dir input/in-the-wild_example\
    --output_dir output/in-the-wild_example
```

## 🏋️ Training

Based on the previously created environment, install extended requirements:

```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```

Set environment parameters for the data directory:

```bash
export BASE_DATA_DIR=YOUR_DATA_DIR  # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) into `${BASE_CKPT_DIR}`
Download the datasets into '${BASE_DATA_DIR}': 
[10-12](https://www.dropbox.com/scl/fi/ghzxygjij7u75awylcli3/10-12.tar?rlkey=78cddl1ogfdpnj0ca7cbdxyj0&st=8ambjjyp&dl=0) 
[10-20](https://www.dropbox.com/scl/fi/xiu3ud3tnbbvna139nh7m/10-20-final.tar?rlkey=u8p9t1nq7zy49sia42a3k31ud&st=mjkod9dt&dl=0) 

Run training script

```bash
python test_train.py --config config/train_collocated_10_20.yaml
```

Resume from a checkpoint, e.g.

```bash
python test_train.py --resume_run output/train_collocated_10_20/checkpoint/latest
```


