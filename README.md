# Training the Untrainable: Introducing Inductive Bias via Representational Alignment
### [Project Page](https://untrainable-networks.github.io) | [Paper]()

[Vighnesh Subramaniam](https://vsubramaniam851.github.io/),
[David Mayo](http://david-mayo.com/),
[Colin Conwell](https://colinconwell.github.io/),
[Tomaso Poggio](https://poggio-lab.mit.edu/people/tomaso-poggio),
[Boris Katz](https://people.csail.mit.edu/boris/boris.html),
[Brian Cheung](https://briancheung.github.io/),
[Andrei Barbu](http://0xab.com/)

This is the implementation of our paper "Training the Untrainable: Introducing Inductive Bias via Representational Alignment". We introduce a method called *Guidance* where we guide an untrainable network usng representational alignment (via centered-kernel alignment) during supervised training. We include two sets of experiments in the paper: (1) fully-connected networks and deep convnets trained on ImageNet and (2) transformers/RNNs trained for sequence modeling.

## Installation and Setup
We recommend creating a new conda/pip environment. We use python 3.9 and pytorch 2.1.1 for all experiments in the paper -- you can install pytorch [here](https://pytorch.org). Afterwards, run
```
pip install -r requirements.txt
```

To run experiments, we used 4 H100 GPUs. 

## Guidance Overview
To see an example of guidance between two networks and substitute your own, we include an overview notebook (`notebooks/guidance_overview.ipynb`) as well as make it a Google Colab [notebook](https://colab.research.google.com/drive/1jxDeRZzhh5vozcbk_WNg86k11R1ja_3Q?usp=sharing). Feel free to subtitute your own metrics outside of CKA!

## Image Classification Experiments
To run experiments with ImageNet, run
```
python -m imagenet-comparison.image_class --exp_name [EXP_NAME] --student_model [STUDENT_MODEL] --batch_size [BATCH_SIZE] --lr [LEARNING_RATE] --num_epochs [NUM_EPOCHS] [--rep_sim] --target_model [TARGET_MODEL] [--untrained]
```

where `--rep_sim` is the flag to indicate using representational alignment and `--untrained` is the flag to indicate whether a guide network should be randomly initialized.

## Sequence Modeling Experiments
To run experiments with a sequence modeling task, run
```
python -m language_modeling.language_modeel --exp_name [EXP_NAME] --task [TASK] --student_model [STUDENT_MODEL] --batch_size [BATCH_SIZE] --lr [LEARNING_RATE] --num_epochs [NUM_EPOCHS] [--rep_sim] --target_model [TARGET_MODEL] [--untrained]
```

where `--task` refers to the specific task e.g. next-word, copy-paste, or parity. Note that specific target networks have been designed for these tasks.