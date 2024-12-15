#!/bin/bash

# Update package lists and install the required system libraries
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1
