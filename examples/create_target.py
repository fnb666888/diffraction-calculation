"""Create an asymmetric test image; no external image input needed."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

directory = Path(__file__).resolve().parent
amplitude = Image.new("L", (128, 96))
draw = ImageDraw.Draw(amplitude)
draw.rectangle((18, 14, 35, 80), fill=220)
draw.rectangle((18, 14, 89, 30), fill=255)
draw.rectangle((35, 40, 70, 54), fill=180)
draw.ellipse((83, 58, 111, 86), fill=140)
amplitude.save(directory/"amplitude_target.png")
y, x = np.mgrid[-1:1:96j, -1:1:128j]
gray = np.clip(.15+.6*np.exp(-((x+.3)**2+(y-.1)**2)/.25)+.18*x, 0, 1)
Image.fromarray((gray*255).astype(np.uint8)).save(directory/"phase_target.png")
print(directory/"amplitude_target.png")
print(directory/"phase_target.png")
