#!/bin/bash

NUM_CLIENTS=1

CLIENT_DIR="/Users/ritvikasonawane/Documents/r24k8xbv/0S25/QFL/quba_fl"

CLIENT_SCRIPT="client.py"

CONDA_ENV="flowerfl"

SERVER_IP=$(ipconfig getifaddr en0)
SERVER_ADDRESS="${SERVER_IP}:8080"

for ((i=0; i<NUM_CLIENTS; i++))
do
    osascript -e "tell application \"Terminal\" to do script \"clear && cd $CLIENT_DIR && conda activate $CONDA_ENV && python $CLIENT_SCRIPT --cid=$i --algorithm FedAVG\""
    sleep 1
done