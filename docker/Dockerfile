FROM nvcr.io/nvidia/pytorch:23.01-py3
MAINTAINER Hristo Buyukliev

RUN pip install einops swig
RUN apt-get update
RUN apt-get install -y tmux nano

# install faiss
RUN git clone https://github.com/facebookresearch/faiss.git
WORKDIR /workspace/faiss
RUN cmake -B build .
RUN make -C build -j faiss
RUN make -C build -j swigfaiss
RUN (cd build/faiss/python && python setup.py install)

VOLUME /HandM_recommend
WORKDIR /HandM_recommend