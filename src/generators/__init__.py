from src.generators.giraffe import GiraffeGenerator
from src.generators.mvcgan import MVCGANGenerator
from src.generators.pigan import PiGANGenerator
from src.generators.eg3d import EG3DGenerator
from src.generators.stylegan2 import StyleGAN2Generator

__init__ = [
    PiGANGenerator,
    GiraffeGenerator,
    MVCGANGenerator,
    EG3DGenerator,
    StyleGAN2Generator,
]
