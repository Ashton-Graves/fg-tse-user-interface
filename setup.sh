#!/bin/bash
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update && sudo apt install -y ffmpeg
fi
pip install -r requirements.txt
