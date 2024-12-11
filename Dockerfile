FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2_560.35.03_linux.run && sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post3 && \
    pip install opencv-contrib-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    git clone -b dev https://github.com/camenduru/MMAudio /content/MMAudio && cd /content/MMAudio && pip install -e . && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/bigvgan_generator.pt -d /content/MMAudio/weights/bigvgan_v2_44khz_128band_512x -o bigvgan_generator.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/raw/main/config.json -d /content/MMAudio/weights/bigvgan_v2_44khz_128band_512x -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k_v2.pth -d /content/MMAudio/weights -o mmaudio_large_44k_v2.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth -d  /content/MMAudio/ext_weights -o synchformer_state_dict.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth -d /content/MMAudio/ext_weights -o v1-44.pth
    
COPY ./worker_runpod.py /content/MMAudio/worker_runpod.py
WORKDIR /content/MMAudio
CMD python worker_runpod.py