# pitnet-anat-public

This code is supplementary to the paper "A Multi-task Network for Anatomy Identification in Endoscopic Pituitary Surgery" and is based on https://github.com/qubvel-org/segmentation_models.pytorch, please cite these works if you find it useful.

```bash
@inbook{Das2023,
  title = {A Multi-task Network for Anatomy Identification in Endoscopic Pituitary Surgery},
  ISBN = {9783031439964},
  ISSN = {1611-3349},
  url = {http://dx.doi.org/10.1007/978-3-031-43996-4_45},
  DOI = {10.1007/978-3-031-43996-4_45},
  booktitle = {Medical Image Computing and Computer Assisted Intervention – MICCAI 2023},
  publisher = {Springer Nature Switzerland},
  author = {Das,  Adrito and Khan,  Danyal Z. and Williams,  Simon C. and Hanrahan,  John G. and Borg,  Anouk and Dorward,  Neil L. and Bano,  Sophia and Marcus,  Hani J. and Stoyanov,  Danail},
  year = {2023},
  pages = {472–482}
}

@misc{Iakubovskii:2019,
  Author = {Pavel Iakubovskii},
  Title = {Segmentation Models Pytorch},
  Year = {2019},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}

```

The code follows the usual PEP8 standards, along with strong typing as outlined in PEP484 and PEP586 where deemed appropriate. The type of a variable is also found at the start of the variable name for ease of reading. Within the scripts directory, utils*.py files contain the data and neural network initialisations, whereas run*.py execute this code for training and evaluation.

The dataset is not publicly available.

The trained model is not publicly available.

I am always open to code review, so please put in push request if you spot any errors or see more efficient/aesthetic ways of presenting the code.

If you have further questions, please do not hesitate to contact Adrito Das on: adrito.das.20@ucl.ac.uk.

Last edited: 2024-10-17.
