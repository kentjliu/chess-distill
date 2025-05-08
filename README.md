# chess-distill

Install dependencies
 Python 3.10.12
`pip install -r requirements.txt`

Install Stockfish 
  For Mac: brew install stockfish
  For Ubuntu users: sudo apt install stockfish

Download and extract data from online (~7.31 GB for 22,590,716	games)
`python extract_data.py`
Note: You can (and probably) edit the script to only download a subset of data. Otherwise, it might either be too much data for your machine or take too long.

Prepare training data in UCI format
`prepare_data.py`

Train policy model
run `policy_model.py`
This model will output a model output a model named 'policy_model.pth' to the models folder

Train self-play model
  run 'AlphaZeroChess960.py'
  This will output models called 'AZ_Model_{it}.pth' to the models folder. We train for 100 iterations and save every 5th model

Distill model
  before running make sure to change the teach model to the path of the 'AZ_Model_{it}.pth' that you hope to use
  run `student_model.py`
  this outputs a model for each epoch and the final distilled model known as 'final_student_model.pth'

for running play_gui.py or eval.py ensure you have put in the correct model you want to test.
Furthermore, ensure that you update the path for stockfish. This can be accessed by writing 
  which stockfish in the CL
  
