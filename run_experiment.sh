#!/bin/bash

# Define the arguments
ARGS="-param params_pendulum1D_samples -env 0 -i 42 -env_model pendulum"

# Run main.py
echo "Running main.py with args: $ARGS"
python3 sampling-gpmpc/main.py $ARGS
if [ $? -ne 0 ]; then
    echo "Error: main.py failed"
    exit 1
fi

# Run visu_car_main.py
echo "Running visu_car_main.py with args: $ARGS"
python3 sampling-gpmpc/visu_car_main.py $ARGS
if [ $? -ne 0 ]; then
    echo "Error: visu_car_main.py failed"
    exit 1
fi

echo "Both scripts executed successfully."