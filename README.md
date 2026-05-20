# Structure Tensor Image Analysis Toolkit

A Python toolkit for analysing image orientations using **structure tensors**.  
This project was developed as part of a bachelor thesis focused on computational image analysis of 2D image data.

The repository contains:

- A reusable `ImageAnalyzer` class
- Example Jupyter notebooks
- Example images for experimentation
- Visualisation tools for orientation analysis
- Histogram and segmentation utilities
- Google Colab compatible workflows

The toolkit is designed to be educational and beginner-friendly for users with limited prior experience in image analysis.

---

# Features

The `ImageAnalyzer` class supports:

## Structure Tensor Analysis
- Gaussian derivative gradients
- Structure tensor computation
- Tensor smoothing
- Eigenvalue and eigenvector decomposition
- Dominant orientation extraction

## Orientation Analysis
- Orientation overlays
- Orientation vector fields
- Orientation histograms
- Polar histograms
- Orientation binning and segmentation
- Alignment percentage computation

## Energy & Anisotropy Tools
- Structure tensor energy computation
- Anisotropy estimation
- Energy masking
- Weighted orientation analysis

## Visualisation Utilities
- HSV orientation overlays
- Dominant orientation vector plots
- Weighted histograms
- Polar histogram plots

---

# Repository Structure

```text
Bachelor_Project/
│
├── Image_analysis_class_week13.py
├── notebooks/
│   ├── example_notebook_1.ipynb
│   ├── example_notebook_2.ipynb
│   └── ...
│
├── images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/danieldidriksen/Bachelor_Project.git
cd Bachelor_Project
```

Install required packages:

```bash
pip install numpy matplotlib scipy
```

---

# Running the Notebooks

The included notebooks are intended as interactive examples and demonstrations of the toolkit.

They can be run:

- In VS Code
- In Google Colab

## Open in Google Colab

The included notebooks can be opened directly in Google Colab using the **"Open in Colab"** button located at the top of each notebook.

This allows the notebooks to be run directly in the browser without requiring a local Python installation.

To use the notebooks in Colab:

1. Open the notebook through the provided Colab button
2. Upload the `Image_analysis_class_week13.py` file
3. Upload any required example images
4. Run the notebook cells normally

## Google Colab Requirements

To ensure that the notebooks run correctly in Google Colab, it is important to use the following package versions:

- `matplotlib==3.10.9`
- `matplotlib-inline==0.2.2`

These can be installed directly in a Colab notebook using:

```python
!pip install --upgrade --force-reinstall matplotlib matplotlib-inline
```

If plotting issues occur, restarting the Colab runtime after installation is recommended.

# Example Workflow

```python
from Image_analysis_class_week13 import ImageAnalyzer

# Load image
img = ImageAnalyzer.load_grayscale("image.png")

# Compute structure tensor
S = ImageAnalyzer.compute_structure_tensor(
    img,
    sigma_derivative=1,
    sigma_tensor=3
)

# Compute anisotropy
anisotropy = ImageAnalyzer.compute_anisotropy_masked(
    S,
    percentile=45
)

# Plot orientation vectors
ImageAnalyzer.plot_orientation_vectors(img, S)

# Plot orientation overlay
theta = ...
ImageAnalyzer.plot_orientation_overlay(
    img,
    theta,
    anisotropy
)
```

---

# Core Concepts

The project is based on the use of **structure tensors** for extracting local orientation information in images.

Using local image gradients, the structure tensor allows estimation of:

- Dominant orientation
- Edge direction
- Local anisotropy

This makes the toolkit useful for analysing:
- Fibrous materials
- Ceramics
- Biological structures
- Textures
- Directional patterns
- Satellite imagery

---

# Included Visualisations

The toolkit includes methods for visualising:

## Orientation Vector Fields
Displays dominant local orientations as vector arrows.

## Orientation Overlay
Uses HSV colouring to visualise orientations directly on the image.

## Orientation Histograms
Shows the distribution of dominant orientations.

## Polar Histograms
Displays orientation distributions in circular form while respecting axial symmetry.

---

# Educational Purpose

This repository was created with a strong focus on readability and learning.

The code is intentionally structured into:
- Low-level helper methods
- Composite functions
- Visualisation utilities

to make the workflow easier to understand and extend.

---

# Dependencies

Main dependencies:

- NumPy
- Matplotlib
- SciPy

---

# Authors

Developed as part of a bachelor project in General Engineering (Cyber Systems specialization) at the Technical University of Denmark (DTU) by Daniel Didriksen and Simon Dubois.

---

# License

This project is intended for educational and research purposes.