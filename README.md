# ACE

## Installation
1. clone repository

`git clone https://github.com/CSUBioGroup/ACE-main.git`

`cd ACE-main/`

2. create env

`conda create -n ACE python=3.8.3`

`conda activate ACE`

3. install pytorch (our test-version: torch==1.12.1+cu116)

`pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116`

4. install other dependencies

`pip install -r requirements.txt`

5. setup

`python setup.py install`
