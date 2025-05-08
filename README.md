# chess-distill

Install dependencies
`pip install -r requirements.txt`

Download and extract data from online (~7.31 GB for 22,590,716	games)
`python extract_data.py`
Note: You can (and probably) edit the script to only download a subset of data. Otherwise, it might either be too much data for your machine or take too long.

Prepare training data in UCI format
`prepare_data.py`

Train policy model
`policy_model.py`

Distill model

`student_model.py`
