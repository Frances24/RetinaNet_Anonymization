#!/usr/bin/env python

import click
import numpy as np
from scipy.stats import poisson

from pathlib import Path
from PIL import Image
from PIL import ImageFilter
import os

arg = click.argument
opt = click.option


IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')

anonymization_methods = {}


def anonymization_method(name):
    def register(f):
        anonymization_methods[name] = f
        return f
    return register


def load_facefile(facefile):
    with open(facefile, 'r') as f:
        faces = []
        for line in f:
            x0, y0, x1, y1 = [int(x) for x in line.split(' ')]
            faces.append((x0, y0, x1, y1))
        return faces


@anonymization_method('blur')
def blur_face(input_image, output_image, facebox):
    facecrop = input_image.crop(facebox)
    w, _ = facecrop.size
    blur_radius = w / 8
    facecrop = facecrop.filter(ImageFilter.GaussianBlur(blur_radius))
    x0, y0, x1, y1 = facebox
    output_image[y0:y1, x0:x1, ...] = facecrop
    

@anonymization_method('zero')
def fill_face(input_image, output_image, facebox, value=0):
    x0, y0, x1, y1 = facebox
    output_image[y0:y1, x0:x1, ...] = value


@anonymization_method('blank')
def blank_face(input_image, output_image, facebox):
    facecrop = input_image.crop(facebox)
    pix_mean = np.array(facecrop).mean(axis=0).mean(axis=0)
    fill_face(input_image, output_image, facebox, pix_mean)


@anonymization_method('pixelate')
def pixelate_face(input_image, output_image, facebox, n=6):
    facecrop = input_image.crop(facebox)
    w, h = facecrop.size
    facecrop = facecrop.resize((n, n), resample=0)
    facecrop = facecrop.resize((w, h), resample=0)
    x0, y0, x1, y1 = facebox
    output_image[y0:y1, x0:x1, ...] = facecrop


@anonymization_method('inpaint')
def inpaint_face(input_image, output_image, facebox):
    image = np.array(input_image)
    mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.bool)  # pylint: disable=E1136
    x0, y0, x1, y1 = facebox  # pylint: disable=E1136
    mask[y0:y1, x0:x1] = 1
    result = poisson.inpaint(image, mask)
    output_image[:] = result


def anonymize_face(input_image, output_image, facebox, method):
    anonymize = anonymization_methods[method]
    anonymize(input_image, output_image, facebox)
    

def process_image(image_in, facefile, image_out, method):
    im = Image.open(image_in)
    faces = load_facefile(facefile)
    processed_image = np.array(im)
    for facebox in faces:
        facebox = clip_box(facebox, im)
        anonymize_face(im, processed_image, facebox, method)
    im = Image.fromarray(processed_image)
    im.save(image_out)


def find_images(path):
    for f in path.iterdir():
        if f.is_file() and f.suffix in IMAGE_SUFFIXES:
            yield f


def clip_box(box, image):
    w, h = image.size
    x0, y0, x1, y1 = box
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, w)
    y1 = min(y1, h)
    return x0, y0, x1, y1


@click.group()
def cli():
    pass


@cli.command()
@arg('image_in')
@arg('facefile')
@arg('image_out')
@opt('--method', default='blur')
def image(image_in, facefile, image_out, method):
    """Blur faces in a image"""
    process_image(image_in, facefile, image_out, method)


@cli.command()
@arg('input_path')
@arg('output_path')
@opt('--method', default='blur')
def folder(input_path, output_path, method):
    """Blur faces in all images in a folder"""
    input_path = Path(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = Path(output_path)
    for imagefile in find_images(input_path):
        print(imagefile)
        facefile = input_path / (imagefile.stem + '.txt')
        if not facefile.is_file():
            continue
        image_out = output_path / imagefile.name
        process_image(imagefile, facefile, image_out, method)


@cli.command()
def methods():
    methods = list(anonymization_methods.keys())
    methods.sort()
    print(*methods, sep='\n')


if __name__ == "__main__":
    cli() 
