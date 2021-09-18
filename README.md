# phodexr

### Project Overview

Phodexr is a project that aims to enable searching through images with text. However, the search should be based solely on the content of the image and not the text context around it. To tackle this problem, phodexr uses CLIP which is a model released by OpenAI to connect two different spaces.

### Repository

This repository has two major folders, app and ml. app contains all the code for the final deliverable which is a web app and ml contains all the ml training code. It is structured this way so everyone can see what other teams are doing.

```
phodexr/
├─ app/
├─ ml/
│  ├─ dataloaders/
│  ├─ training_functions/
│  ├─ models/
```

#### Repository Setup

We will use virtual environments to help isolate all the packages and versions used by this project. The following steps walk through creating and using a virtual environment.

1. Create the virutal environment by running `python3 -m venv ./venv`. This creates a new folder called venv that houses your virtual environment.
2. Activate the virutal environment by running `source ./venv/bin/activate`. You are now inside an environment isolated from your global python installation.
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Login to weights and biases via `wandb login`. Follow instructions to setup API key.
5. Allow `run.sh` to be executed by running `chmod +x run.sh`
6. Verify installation is working by running `./run.sh --epochs 1 train_cnn_cifar10`.
7. Deactivate virtual environment by running `deactivate`.

If there is a problem with pycocotools, you can try these steps. (Currently blind trusting pycocotools on pip which might cause problems)

1. Verify that cython is already installed via `pip freeze | grep -i cython`
2. Install pycocotools via `pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`

#### App Folder

The app folder contains a react app boostrapped using create-react-app with the TypeScript template. This will house the code for the final deliverable.

#### ML Folder

The layout of the ml repository is such that you can add code for your training loops and run it with `run.sh`.

It is designed this way so there is a clearer layout of the code and everyone can look at anyone else's code and understand what is happening.

To add to the repository, you would add relevant code to each of the subfolders. For example, to implement a CNN trained on MNIST, I would add a MNIST dataloader to the dataloaders folder, a CNN model to the models folder, and finally a training loop that accepts various parameters to the training_functions folder. Finally, you can add the training function to `__main__.py` to accept command line arguments.

For example, to run the example CNN model I would run `./run.sh --epochs=3 --lr=0.001 train_cnn`