# CS354 Spring-21 Group Project
Team project for CS354 (spring 2021): Classification of networks based on limited samples using GNNs.

### Getting started.
# Requirements
Run install_requirements.bash in order to set up all dependancies.

# Set up data
Either
1. Download processed RDS samples from: 
2. Download unprocessed data from https://snap.stanford.edu/data/enwiki-2013.html. Then create RDS samples using make_data.graph2samp.py

# Run
Run main.py with a command arg specifying the path of your settings file to begin training.
To use our settings run "python main.py grain_training.settings"

