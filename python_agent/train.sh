#!/bin/bash
#source ~/miniconda/bin/activate
echo "Script executed from: ${PWD}"
TF_CPP_MIN_LOG_LEVEL=3 python python_agent/src/mp_self_play.py