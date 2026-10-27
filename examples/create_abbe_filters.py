"""Deterministic demonstration masks; their centres represent the DC sample."""
from pathlib import Path
import numpy as np
from PIL import Image


def create(directory=None):
    directory=Path(directory) if directory else Path(__file__).resolve().parent
    directory.mkdir(parents=True,exist_ok=True)
    x=(np.arange(256)-128)/256
    X,Y=np.meshgrid(x,x)
    amplitude=np.exp(-(X*X+Y*Y)/(2*.2**2))*(.25+.75*(np.abs(Y)<.08))
    phase=.5+.35*np.sin(2*np.pi*X)*np.cos(2*np.pi*Y)
    for values,name in ((amplitude,"abbe_filter_amplitude.png"),(phase,"abbe_filter_phase.png")):
        image_values=np.roll(values[::-1],1,axis=0)
        Image.fromarray(np.rint(image_values*255).astype(np.uint8)).save(directory/name)


if __name__=="__main__":
    create()
