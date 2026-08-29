# Build a Text-to-Image Generator (from Scratch) — Companion

A follow-along workspace for **Mark Liu's _Build a Text-to-Image Generator (from Scratch)_** (Manning, 2026).

## Purpose

Read each chapter from your local copy, then implement the material in its matching notebook. Each notebook intentionally starts with only a title and short introduction.

The personally licensed source PDF is **not included** in this public repository. The author's executable examples are available from the [official code repository](https://github.com/markhliu/txt2img).

## Setup

```bash
uv sync
uv run jupyter lab
```

### CUDA-enabled PyTorch

`pyproject.toml` resolves PyTorch and TorchVision from the official CUDA 13.0 (`cu130`) wheel index. Verify the active environment with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The project declares the libraries used across the book, including PyTorch's vision stack, Diffusers, Transformers, Datasets, spaCy, image-processing packages, and notebook tooling. Chapter-specific external repositories such as OpenAI CLIP, `taming-transformers`, Stable Diffusion, and min-DALL-E should be followed from the book rather than vendored here.

## Shared terminology

Keep canonical vocabulary in [`glossary.md`](glossary.md).

## Reusing notebook-defined code

This repository uses [nbdev](https://nbdev.fast.ai/) to export selected notebook cells into importable modules. After changing any `#| export` cell, run:

```bash
uv run nbdev-export
```

Files under `build_a_text_to_image_generator_from_scratch_companion/` are generated from notebooks and should not be edited manually.

## Chapters

| # | Chapter | Notebook | Issue |
|---:|---|---|---|
| 1 | A tale of two models: Transformers and diffusions | [Notebook](notebooks/01-a-tale-of-two-models-transformers-and-diffusions.ipynb) | [#1](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/1) |
| 2 | Build a transformer | [Notebook](notebooks/02-build-a-transformer.ipynb) | [#2](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/2) |
| 3 | Classify images with a vision transformer | [Notebook](notebooks/03-classify-images-with-a-vision-transformer.ipynb) | [#3](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/3) |
| 4 | Add captions to images | [Notebook](notebooks/04-add-captions-to-images.ipynb) | [#4](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/4) |
| 5 | Generate images with diffusion models | [Notebook](notebooks/05-generate-images-with-diffusion-models.ipynb) | [#5](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/5) |
| 6 | Control what images to generate in diffusion models | [Notebook](notebooks/06-control-what-images-to-generate-in-diffusion-models.ipynb) | [#6](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/6) |
| 7 | Generate high-resolution images with diffusion models | [Notebook](notebooks/07-generate-high-resolution-images-with-diffusion-models.ipynb) | [#7](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/7) |
| 8 | CLIP: A model to measure the similarity between image and text | [Notebook](notebooks/08-clip-image-text-similarity.ipynb) | [#8](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/8) |
| 9 | Text-to-image generation with latent diffusion | [Notebook](notebooks/09-text-to-image-generation-with-latent-diffusion.ipynb) | [#9](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/9) |
| 10 | A deep dive into Stable Diffusion | [Notebook](notebooks/10-a-deep-dive-into-stable-diffusion.ipynb) | [#10](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/10) |
| 11 | VQGAN: Convert images into sequences of integers | [Notebook](notebooks/11-vqgan-convert-images-into-sequences-of-integers.ipynb) | [#11](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/11) |
| 12 | A minimal implementation of DALL-E | [Notebook](notebooks/12-a-minimal-implementation-of-dall-e.ipynb) | [#12](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/12) |
| 13 | New developments and challenges in text-to-image generation | [Notebook](notebooks/13-new-developments-and-challenges.ipynb) | [#13](https://github.com/majorgilles/build-a-text-to-image-generator-from-scratch-companion/issues/13) |

## Local source

The companion was outlined from:

```text
C:\Users\giloz\Downloads\Build_a_Text-to-Image_Generator_(from_Sc.pdf
```

The appendix is intentionally excluded.
