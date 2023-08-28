# project-DD

<a href="https://colab.research.google.com/github/datvodinh10/project-DD/blob/master/main.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Model

![Alt text](img/model.png)

## Repo Structure

```
│   .gitignore
│   a.ipynb
│   LICENSE
│   main.ipynb
│   README.md
│
├───demo
├───img
│       model.png
│
└───src
    ├───backbone
    │   │   resnet.py
    │   │   swin_transformer.py
    │   │   vgg.py
    │   │   ViT.py
    │   └───
    ├───model
    │   │   model.py
    │   │   trainer.py
    │   └───
    └───utils
        │   generator.py
        │   inference.py
        │   transform.py
        │   vocab.py
        │   writer.py
        └───
```