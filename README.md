# GP_test1
graduation project test1
## Setup
Install anaconda
```Shell
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh
source ~/.profile
```

Clone the repo and build the environment
```Shell
git clone https://github.com/FrankGu0911/GP_test1
cd GP_test1
conda create -n GP_test1 python=3.7
conda activate GP_test1
pip3 install -r requirements.txt
```

Download and setup CARLA 0.9.10.1
```Shell
chmod +x setup_carla.sh
bash setup_carla.sh
```