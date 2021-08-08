ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.03-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /tmp/torchaudio
RUN git clone --depth 1 --branch release/0.7 https://github.com/pytorch/audio.git && \
    cd audio && \
    BUILD_SOX=1 python setup.py install && \
    cd .. && rm -r audio

ENV PYTHONPATH /workspace/cpc
WORKDIR /workspace/cpc

RUN conda install -y \
    tqdm \
    psutil \
    openblas-devel \
    nose \
    cython

ADD requirements.txt .
RUN pip install -r requirements.txt
