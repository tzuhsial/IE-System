# ImageEditingDialogueSelfPlay

This repository uses the **Machines Talking to Machines(M2M)** framework to generate synthetic dialogues tailored to the image edit domain.  A brief outline of the framework is provided below. For further details, please take a look at their [paper](https://arxiv.org/pdf/1801.04871.pdf).

### M2M framework

1. The dialogue developer provides a task schema
2. Automated bots generate dialogue outlines
3. Crowd workers rewrite the utterances and violate slot spans.
4. A dialogue model is trained with supervised-learning on the collected dataset.


### Image Editing Task Schema



### UserBot and SystemBot


### Getting Started


### Installation

```bash
pip install -r requirementst.txt
```

### TODO
1. adjust slots
2. non_adjust slots
3. integration with photoshop api

