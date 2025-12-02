import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def random_letter(pp, letter, ii, position):
    _, rows, columns = pp.size()
    image_size = (columns, rows)
    image = Image.new("RGB", image_size, "black")
    font = ImageFont.truetype(r'./util/arial.ttf', 80)
    draw = ImageDraw.Draw(image)    
    position = (position, 80*ii-5)
    transform = transforms.ToTensor()
    draw.text(position, letter, fill="white", font=font)
    gray_image = image.convert("L")
    img = np.flipud(np.array(gray_image))
    img = transform(img.copy()).to(torch.complex64)
    return img

def process_p_k(p_k, k, nb_perturbations):
    positions = [20,110,200,290]
    M = 5e-2
    phases = np.linspace(0, np.pi, num=nb_perturbations, endpoint=False)
    if k % 6 == 0:
        multiplier = 1
    elif k % 6 == 1:
        multiplier = -1
    elif k % 6 == 2:
        multiplier = 0.25
    elif k % 6 == 3:
        multiplier = -0.25
    elif k % 6 == 4:
        multiplier = 0.5
    else:
        multiplier = -0.5

    letter_intensity = {
        i: (letter, M * np.cos(theta) + 1j * np.abs(M * np.sin(theta)) * multiplier)
        for i, (letter, theta) in enumerate(zip(['A', 'R', 'O', 'N', 'P', 'B', 'Z', 'L'], phases))
    }

    if k in letter_intensity:
        letter, intensity = letter_intensity[k]
        for ii in range(len(positions)):
            rand_letter = random_letter(p_k, letter, ii, positions[ii])
            p_k[abs(rand_letter) > 0.0] = torch.tensor(intensity, dtype=torch.complex64)
    else:
        raise ValueError(
            f"Too many masks! # of perturbation masks = {k+1} is out of the expected range "
            f"(1-{len(letter_intensity)}). Please redefine some of the intensities in perturbation_utils.py file."
        )
    return p_k