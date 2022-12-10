# LatentSwap3D: Semantic Edits on 3D Image GANs
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  [![Edit In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/enisimsar/latentswap3d/blob/main/demo/MVCGAN/FFHQ_sample.ipynb)
  [![Edit Real Face In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/enisimsar/latentswap3d/blob/main/demo/MVCGAN/FFHQ_inversion.ipynb)
  
> Recent 3D-aware GANs rely on volumetric rendering techniques to disentangle the pose and appearance of objects, de facto generating entire 3D volumes rather than single-view 2D images from a latent code. Complex image editing tasks can be performed in standard 2D-based GANs (e.g., StyleGAN models) as manipulation of latent dimensions. However, to the best of our knowledge, similar properties have only been partially explored for 3D-aware GAN models. This work aims to fill this gap by showing the limitations of existing methods and proposing LatentSwap3D, a model-agnostic approach designed to enable attribute editing in the latent space of pre-trained 3D-aware GANs. We first identify the most relevant dimensions in the latent space of the model controlling the targeted attribute by relying on the feature importance ranking of a random forest classifier. Then, to apply the transformation, we swap the top-K most relevant latent dimensions of the image being edited with an image exhibiting the desired attribute. Despite its simplicity, LatentSwap3D provides remarkable semantic edits in a disentangled manner and outperforms alternative approaches both qualitatively and quantitatively. We demonstrate our semantic edit approach on various 3D-aware generative models such as pi-GAN, GIRAFFE, StyleSDF, MVCGAN, EG3D and VolumeGAN, and on diverse datasets, such as FFHQ, AFHQ, Cats, MetFaces, and CompCars.


## Geting Started

### Installation

Install the dependencies in ``env.yml``
``` bash
$ conda env create -f env.yml
$ conda activate latent-3d
```

### Quickstart

For a quick demo, see [DEMO](demo/MVCGAN).

- FFHQ Random Face Example: [Edit In Colab](http://colab.research.google.com/github/enisimsar/latentswap3d/blob/main/demo/MVCGAN/FFHQ_sample.ipynb)
- FFHQ Real Face Example: [Edit Real Face In Colab](http://colab.research.google.com/github/enisimsar/latentswap3d/blob/main/demo/MVCGAN/FFHQ_inversion.ipynb)

### Hydra Usage

The repository uses [Hydra](https://hydra.cc) framework to manage experiments.
We provide seven main experiments:

- ``gen.py``: Generates the dataset used to find feature ranking.
- ``predict.py``: Predicts the pseudo-labels of the generated dataset.
- ``dci.py``: Calculates the DCI metrics for the candidate latent spaces.
- ``find.py``: Finds the feature ranking for attributes.
- ``tune.py``: Tunes the parameter K, number of dimensions, that will be swapped.
- ``manipulate.py``: Applies the semantic edits on random samples.
- ``encode.py``: Encodes the real face image.

Hydra will output experiment results under ``outputs`` folder.

### Example for MVCGAN

``` bash
python gen.py hparams.batch_size=1 num_samples=10000 generator=mvcgan generator.class_name=FFHQ
OUTPUT_PATH=outputs/run/src.generators.MVCGANGenerator/FFHQ/2022-11-23
python predict.py hparams.batch_size=50 load_path=$OUTPUT_PATH generator=mvcgan generator.class_name=FFHQ
python dci.py load_path=$OUTPUT_PATH generator=mvcgan generator.class_name=FFHQ
python find.py load_path=$OUTPUT_PATH generator=mvcgan generator.class_name=FFHQ
python tune.py load_path=$OUTPUT_PATH generator=mvcgan generator.class_name=FFHQ
python manipulate.py load_path=$OUTPUT_PATH generator=mvcgan generator.class_name=FFHQ
```
## Citation

If you use this code for your research, please cite our paper:
```
@misc{simsar2022latentswap3d,
  doi = {10.48550/ARXIV.2212.01381},
  url = {https://arxiv.org/abs/2212.01381},
  author = {Simsar, Enis and Tonioni, Alessio and Örnek, Evin Pınar and Tombari, Federico},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {LatentSwap3D: Semantic Edits on 3D Image GANs},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
