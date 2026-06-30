# Third-Party Software And Models

This repository does not include third-party model weights or vendored copies of large external projects.

## Depth Anything 3

The panorama and video workflows use Depth Anything 3 for image-based visible-depth estimation. To run those workflows, install or clone Depth Anything 3 separately and pass its path with `--da3-repo`.

The DA3 model weights must also be downloaded separately and passed with `--model-dir`.

Follow the upstream Depth Anything 3 license and model-use terms when using the image-based workflows.

## PyTorch And Scientific Python Packages

Python package dependencies are listed in `requirements.txt` and `requirements-optional.txt`. Their licenses are controlled by the respective upstream projects.

## Data

Point clouds, panorama images, inspection videos, trajectory files, railway centreline data, and generated outputs are not included. These files may be large, restricted, or project-specific.
