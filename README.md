# 🍎 Fruit State Classifier App – Backend

## 💡 Overview

This is the backend of Fruit State Classifier App project. Server was built using Python and FastAPI framework. Application involes usage of machine learning classification model, which is based on resnet34 architecture, fine-tuned to work with kaggle nourabdoun/fruits-quality-fresh-vs-rotte dataset, which includes samples of fruits sorted in two categories: fresh and rotten.

## 🎯 Model Training

Model was trained, with such hyperparameters:
```python
learning_rate = 0.001
batch_size = 32
gamma = 0.95
patience = 5
num_epochs = 20
```

## 🗒️ Features

* 🛜 Easy to use and develop REST API based on FastAPI
* 🤖 Pytorch model, with it's parameters ready to be used in practice
* ⚙️ Docker support for local development
* ⚡ Implementatio of pytorch lightning
* 📈 Tensorboard implementation via pytorch lightning

## ⚙️ Command Tools

To work with this project locally or in a containerized environment, use the following commands:
```bash
conda env export > environment.yml # generates list of dependencies, which are used by conda

sips -s format jpeg ./evaluation_data/[file_name].HEIC --out ./evaluation_data/[file_name].jpg # converting HEIC -> JPG command on MacOS

python -m model.train # model training command

python -m model.eval # model evaluation on your own files

tensorboard --logdir=lightning_logs # tensorboard starting command

uvicorn api.server:app --reload # server development starting command

docker-compose up # 🐳 Run with Docker (backend + frontend)
````

🧠 Tech Stack
* pytorch lightning
* tensorboard
* torchvision

<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,fastapi,pytorch,anaconda,docker" />
  </a>
</p>
 