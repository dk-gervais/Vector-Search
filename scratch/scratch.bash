#!/bin/bash

apt-get update --yes && \
    apt-get install --yes --no-install-recommends git python3 python3-venv pip

mkdir /home/code
cd /home/code
git clone https://github.com/dk-gervais/Vector-Search.git
cd Vector-Search

python3 -m venv /home/env
source /home/env/bin/activate
pip install -r requirements.txt
