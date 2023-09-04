# LatentSwap3D: Semantic Edits on 3D Image GANs
  <a href="https://enis.dev/latentswap3d/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  [![Edit In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/enisimsar/latentswap3d/blob/main/demo/MVCGAN/FFHQ_sample.ipynb)
  [![Edit Real Face In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/enisimsar/latentswap3d/blob/main/demo/MVCGAN/FFHQ_inversion.ipynb)
  
> 3D GANs have the ability to generate latent codes for entire 3D volumes rather than only 2D images. These models offer desirable features like high-quality geometry and multi-view consistency, but, unlike their 2D counterparts, complex semantic image editing tasks for 3D GANs have only been partially explored. To address this problem, we propose LatentSwap3D, a semantic edit approach based on latent space discovery that can be used with any off-the-shelf 3D or 2D GAN model and on any dataset. LatentSwap3D relies on identifying the latent code dimensions corresponding to specific attributes by feature ranking using a random forest classifier. It then performs the edit by swapping the selected dimensions of the image being edited with the ones from an automatically selected reference image. Compared to other latent space control-based edit methods, which were mainly designed for 2D GANs, our method on 3D GANs provides remarkably consistent semantic edits in a disentangled manner and outperforms others both qualitatively and quantitatively. We show results on seven 3D GANs (?-GAN, GIRAFFE, StyleSDF, MVCGAN, EG3D, StyleNeRF, and VolumeGAN) and on five datasets (FFHQ, AFHQ, Cats, MetFaces, and CompCars).


## Geting Started

### Installation

`$ git clone --recurse-submodules -j8 git@github.com:enisimsar/latentswap3d.git`

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
