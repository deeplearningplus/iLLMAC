# iLLMAC: instruction-tuned LLM for Assessment of Cancer

## Introduction
Inspired by the success of large language model in natural language understanding, we herein present a LLM-based model – instruction-tuned LLM for Assessment of Cancer (iLLMAC) – that can detect cancer using cfDNA end-motif profiles. We developed this model with cfDNA sequencing data curated from 2451 individuals. The sequencing modalities include whole genome sequencing, bisulfite sequencing and 5-hydroxymethylcytosine sequencing. We evaluated the performance of the model in the diagnosis of cancer and detection of HCC with internal- and external-testing sets. We demonstrated that iLLMAC is able to achieve high detection accuracy on different modalities of cfDNA data. Besides the development of iLLMAC, our study presents a new paradigm for cfDNA-based cancer diagnosis.   

## System requirements
- Operating systems: CentOS 7.
- [Python](https://docs.conda.io/en/latest/miniconda.html) (version == 3.7).
- [PyTorch](https://pytorch.org) (version == 1.13.1+cu116).
- [transformers](https://huggingface.co/docs/transformers/index) (version == 4.28.1).

This example was tested with the following environment. However, it should work on the other platforms. 

## Installation guide
- Following instruction from [miniconda](https://docs.conda.io/en/latest/miniconda.html) to install Python.
- Use the following command to install required packages.
```bash
# Install with GPU support. Check https://pytorch.org for more information. 
#+The following cmd install PyTorch compiled with cuda 118. 
pip install torch --index-url https://download.pytorch.org/whl/cu118

# If GPU not available, install the PyTorch compiled for CPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install transformers, tokenizers and prettytable
pip install transformers==4.28.1 tokenizers==0.13.3 prettytable
```

- The installation process will take about an hour. This heavily depends on your network bandwidth.

## Demo
- Clone `iLLMAC` locally from Github
```bash
git clone https://github.com/deeplearningplus/iLLMAC.git
gzip -d ./data/*
```
- Instructions to train iLLMAC:
```bash
bash iLLMAC_train.sh
```

The trained model will be saved in `out` when the above command finishes running.
We uploaded a pretrained model in [BaiduDisk](https://markdown.com.cn)(Password:1234) for this tutorial. Download it, and put it in `llama/7b-32` folder.

- Instructions to evaluate iLLMAC:
```bash
bash iLLMAC_predict.sh
```


## How to run on your own data
prepare the training data in the same format as `data/train_data_points-v2.1-64.json` and run `iLLMAC_train.sh`.




