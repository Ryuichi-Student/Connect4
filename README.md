# Connect4
A small showcase project. Originally a project I did to learn TypeScript, but I decided to implement an AlphaZero strategy which ended up being a large chunk of the project. While I haven't tested it against a formal benchmark, it seems to do pretty well, at least vs me!

I also may decide to include a few other entirely unrelated scripts I'm proud of (in the ignore folder).


## How to use:
- create and activate a miniconda environment.
- install the requirements on Conda
- check python_agent/src/mp_self_play.py to set up config variables
- run the following commands in the terminal to train a model:
```
bash python_agent/train.sh  
# use a different terminal to run tensorboard
tensorboard --logdir logs/fit 
```

- Model weights will be saved to python_agent/src/models/checkpoint
- Best model weights will be saved to python_agent/src/models/best_model
- To play against the model, run the following command:
```
python python_agent/src/main.py
```
- To play on a nice web GUI, run the following command:
```
python src/strategies/python_agent_backend.py    
# Again, in a different terminal
http-server
```
- Make any changes to any of the TypeScript files in src/ and run the following command to compile:
```
bash compile.sh
```
## Requirements:
Only tested on M1 Pro Macbook Pro (Apple Silicon) and Python 3.10.9
As mentioned earlier, I've only tried miniconda, with the requirements in requirements.txt.
