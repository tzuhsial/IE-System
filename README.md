# Conversational Image Editing

### Getting Started

To start the system website, you will need
1. [Conversational Image Editing](https://github.com/iammrhelo/ConversationalImageEditing)(this repo)
2. [EditmeTagger](https://github.com/iammrhelo/EditmeTagger)(NLU)
3. [MongoDB](https://docs.mongodb.com/manual/installation/#mongodb-community-edition)(for session and logging purposes)

#### Installation

Create a conda environment and setup with the necessary packages

```bash
# Creates a conda environment named "cie" in python3.5
conda create -n cie python=3.5
conda activate cie
pip install -r requirements.txt
conda install -c menpo opencv3 
```

#### Start service

After starting EditmeTagger and MongoDB on localhost, run the following command to start the service. Current config uses Rule-based Policy.
```bash
./realuser_server.sh
```

### TODO
* Evaluation
    - [x] Set maximum number of turns, or execution
    - [x] Customize goal index instead of random sampling
    - [x] Record selected policy

* User Interface
    - [x] Debug gesture_click on object location image
    - [x] object_mask_str user input should remove system queries
    - [x] Display turn count

* Photoshop
    - [ ] Optimize image processing, this is the current bottleneck
