import numpy as np
import sympy as sp 
import matplotlib.pyplot as plt
import dtumathtools as dtu
import scipy.ndimage as sci 
from matplotlib import cm


class ImageAnalyzer:

# LOADING GRAYSCALE
    @staticmethod
    def load_crop_grayscale(path, crop=None):
        """
        Load image, optionally crop, and convert to grayscale.
        crop = (row_start, row_end, col_start, col_end)
        """
        img = plt.imread(path)
        if crop is not None:
            r1, r2, c1, c2 = crop
            img = img[r1:r2, c1:c2]
        if img.ndim == 3:
            img = img.mean(axis=2)
        return img

#PLOT IMAGE
    @staticmethod
    def plot_image(img, title = "Grayscale Image" ):
        """Display image in grayscale."""
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.show()

# GRADIENTS
    @staticmethod
    def compute_gradients(img, sigma_deriv=1):
        """
        Compute image gradients using Gaussian derivatives.
        Returns Ix, Iy
        """
        Ix = sci.gaussian_filter(img, sigma=sigma_deriv, order=[0, 1])
        Iy = sci.gaussian_filter(img, sigma=sigma_deriv, order=[1, 0])
        return Ix, Iy

#CREATING STRUCTURE TENSOR
    @staticmethod
    def structure_tensor(Ix, Iy):
        """
        Compute raw structure tensor components.
        Returns S11, S12, S22
        """
        S11 = Ix * Ix
        S12 = Ix * Iy
        S22 = Iy * Iy
        return S11, S12, S22

#SMOOTHING STRUCTURE TENSOR
    @staticmethod
    def smooth_tensor(S11, S12, S22, sigma_tensor=2):
        """
        Apply Gaussian smoothing to tensor components.
        """
        S11 = sci.gaussian_filter(S11, sigma=sigma_tensor)
        S12 = sci.gaussian_filter(S12, sigma=sigma_tensor)
        S22 = sci.gaussian_filter(S22, sigma=sigma_tensor)
        return S11, S12, S22
    
#STORING STRUCTURE TENSOR AS MATRIX 
    @staticmethod
    def tensor_to_matrix(S11, S12, S22):
        """
        Stack structure tensor components into (H, W, 2, 2) matrix field.
        """
        S = np.stack([
            np.stack([S11, S12], axis=-1),
            np.stack([S12, S22], axis=-1)
        ], axis=-2)
        return S

#COMPUTE EIGEN-VALUES AND -VECTORS
    @staticmethod
    def eigendecomposition(S):
        """
        Compute eigenvalues and eigenvectors for each pixel tensor.
        Returns:
        evals: (H, W, 2)
        evecs: (H, W, 2, 2)
        """
        evals, evecs = np.linalg.eigh(S)
        return evals, evecs

#FIND LAMBDA SMALL AND LARGE 
    @staticmethod
    def split_eigenpairs(evals, evecs):
        """
        Split eigenvalues and eigenvectors into small and large components.
        """
        lam_small = evals[:, :, 0]
        lam_large = evals[:, :, 1]

        v_small = evecs[:, :, :, 0]
        v_large = evecs[:, :, :, 1]

        return lam_small, lam_large, v_small, v_large

#FIND DOMINANT ANGLE FROM EIGENVECTOR
    @staticmethod
    def orientation_from_eigenvectors(v_small):
        """
        Compute orientation angle from smallest eigenvector.
        """
        theta = np.arctan2(-v_small[:, :, 1], v_small[:, :, 0])
        theta = np.mod(theta, np.pi)
        return theta

#CREATE UNIT VECTOR FOR DOMINANT ORIENTATION
    @staticmethod
    def orientation_to_unit_vectors(theta):
        """
        Convert orientation angle to unit direction vectors.
        """
        vx = np.cos(theta)
        vy = np.sin(theta)
        return vx, vy
    
#COMPUTE TOTAL ENERGY PER PIXEL 
    @staticmethod
    def compute_energy(lam_small, lam_large):
        """
        Compute total structure tensor energy.
        """
        return lam_small + lam_large

#COMPUTE ISOTROPY AND ANISOTROPY
    @staticmethod
    def compute_isotropy_anisotropy(lam_small, lam_large):
        """
        Compute isotropy and anisotropy measures.
        """
        isotropy = lam_small / (lam_large + 1e-10)
        anisotropy = 1 - isotropy
        return isotropy, anisotropy
    
#COMPUTE ENERGY CUTOFF MASK
    @staticmethod
    def energy_mask(energy, percentile=45):
        """
        Create mask for high-energy pixels.
        """
        threshold = np.percentile(energy, percentile)
        return energy >= threshold
    
#APPLY ENERGY CUTOFF TO ANISOTROPY
    @staticmethod
    def mask_anisotropy(anisotropy, mask):
        """
        Zero out anisotropy where energy is low.
        """
        result = anisotropy.copy()
        result[~mask] = 0
        return result
    
#PLOT ORIENTATION OVERLAY WITH CHOSEN ENERGY CUTOFF
    @staticmethod
    def plot_orientation_overlay(image, theta, anisotropy, percentile):
        """
        Plot orientation overlay using precomputed anisotropy.
        """

        theta_norm = theta / np.pi

        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap="gray")
        plt.imshow(
            theta_norm,
            cmap="hsv",
            alpha=0.55 * anisotropy,
            interpolation="nearest"
        )
        plt.axis("off")
        plt.title(f"Top {100 - percentile}% Energy Pixels")
        plt.show()

# ORIENTATION HISTOGRAM
    @staticmethod
    def compute_orientation_histogram(theta, anisotropy_energy, bins=180):
        """
        Compute weighted histogram of orientations.

        Parameters:
        - theta: orientation field (radians, [0, pi])
        - anisotropy_energy: weights (same shape as theta)
        - bins: number of histogram bins

        Returns:
        - counts
        - bin_centers
        - bin_width
        """

        theta_flat = theta.ravel()
        weights_flat = anisotropy_energy.ravel()

        valid = np.isfinite(theta_flat) & np.isfinite(weights_flat)

        theta_flat = theta_flat[valid]
        weights_flat = weights_flat[valid]

        counts, bin_edges = np.histogram(
            theta_flat,
            bins=bins,
            range=(0, np.pi),
            weights=weights_flat
        )

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        return counts, bin_centers, bin_width
    
#PLOTTING WEIGHTED HISTOGRAM 
    @staticmethod
    def plot_orientation_histogram(counts, bin_centers, bin_width):
        """
        Plot orientation histogram with HSV colouring.
        """

        # Normalize angles to [0,1] for colourmap
        bin_centers_norm = bin_centers / np.pi

        # Map each bin to a colour
        bar_colours = plt.cm.hsv(bin_centers_norm)

        plt.figure(figsize=(6, 6))

        plt.bar(
            bin_centers,
            counts,
            width=bin_width,
            color=bar_colours,
            align='center'
        )

        plt.xlabel("Angle")
        plt.ylabel("Weighted pixel count")
        plt.title("Histogram of Dominant Orientations")

        plt.xlim(0, np.pi)

        plt.xticks(
            [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
            ["0","π/4", "π/2", "3π/4", "π"]
        )

        plt.show()

# COMPUTATIONS FOR POLAR HISTOGRAM
    @staticmethod
    def compute_polar_histogram(counts, bin_centers):
        """
        Prepare data for symmetric polar histogram.

        Returns:
        - angles_full
        - counts_full
        - colours_full
        """

        # Duplicate angles to cover [0, 2π]
        angles_full = np.concatenate([bin_centers, bin_centers + np.pi])

        # Duplicate counts (symmetry)
        counts_full = np.concatenate([counts, counts])

        # Colour mapping (based on original angles)
        colours = plt.cm.hsv(bin_centers / np.pi)

        # Duplicate colours
        colours_full = np.concatenate([colours, colours], axis=0)

        return angles_full, counts_full, colours_full

#PLOT POLAR HISTROGRAM 
    @staticmethod
    def plot_polar_histogram(angles_full, counts_full, colours_full, bin_width):
        """
        Plot weighted polar histogram.
        """

        fig, ax = plt.subplots(
            subplot_kw={'projection': 'polar'},
            figsize=(6, 6)
        )

        ax.bar(
            angles_full,
            counts_full,
            width=bin_width,
            color=colours_full,
            align='center'
        )

        # Set angle ticks every 15 degrees
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 15)))

        plt.title("Weighted Polar Histogram")
        plt.show()

# ORIENTATION BINNING
    @staticmethod
    def bin_orientations(theta, centers, tolerance_deg=10):
        """
        Assign each pixel orientation to nearest dominant orientation (if within tolerance).

        Parameters:
        - theta: orientation field (radians, [0, pi])
        - centers: list/array of dominant orientations (degrees)
        - tolerance_deg: angular tolerance per bin (degrees)

        Returns:
        - chosen_theta_binned (same shape as theta, NaN where no match)
        """

        centers = np.array(np.deg2rad(centers))
        tolerance = np.deg2rad(tolerance_deg)
        chosen_theta_binned = np.full_like(theta, np.nan)

        for center in centers:
            # Circular distance (important!)
            diff = np.abs(np.angle(np.exp(1j * (theta - center))))
            mask = diff < tolerance
            chosen_theta_binned[mask] = center
        return chosen_theta_binned
    
#PLOTTING BINNED ORIENTATIONS
    @staticmethod
    def plot_binned_orientations(
        image,
        chosen_theta_binned,
        anisotropy_energy,
        angles_deg,
        tolerance_deg=10
    ):
        """
        Overlay selected orientations on image.

        Parameters:
        - image: grayscale image
        - chosen_theta_binned: output from bin_orientations
        - anisotropy_energy: masked anisotropy weights
        - angles_deg: list of chosen dominant orientations in degrees
        - tolerance_deg: angular tolerance in degrees
        """

        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap="gray")
        plt.imshow(
            chosen_theta_binned,
            cmap="hsv",
            vmin=0,
            vmax=np.pi,
            alpha=0.55 * anisotropy_energy,
            interpolation="nearest"
        )

        plt.title(f"Binned Orientations with Angles: {angles_deg}° and a Tolerance of ±{tolerance_deg}°")
        plt.show()

#COMPUTING ALIGNMENT PERCENTAGE
    @staticmethod
    def compute_alignment_percentage(theta, anisotropy_energy, dominant_angles_deg, tolerance_deg=10):
        """
        Compute percentage of weighted pixels aligned with dominant orientations.

        Returns:
        - percentage (0–100)
        """

        # Convert inputs
        theta_flat = theta.ravel()
        weights = anisotropy_energy.ravel()

        dominant_rads = np.deg2rad(dominant_angles_deg)
        tolerance = np.deg2rad(tolerance_deg)

        # Axial distance (orientation invariant: π-periodic)
        def axial_distance(a, b):
            d = np.abs(a - b) % np.pi
            return np.minimum(d, np.pi - d)

        # Build alignment mask
        alignment = np.zeros_like(theta_flat, dtype=bool)

        for a in dominant_rads:
            alignment |= (axial_distance(theta_flat, a) < tolerance)

        # Weighted percentage
        total_energy = np.sum(weights)
        aligned_energy = np.sum(weights * alignment)

        return 100 * aligned_energy / (total_energy + 1e-12)
    

    @staticmethod
    def print_alignment_percentage(percentile, dominant_angles_deg, tolerance_deg=10):
        """
        print alignment percentage.
        """
        print(
            f"{percentile:.2f}% of orientations align with "
            f"{dominant_angles_deg}° (±{tolerance_deg}°)"
        )