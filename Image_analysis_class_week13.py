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
        Expected to be used on either Grayscale- or RGB images.
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
        return anisotropy
    
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
    def plot_orientation_overlay(image, theta, anisotropy, percentile, opaqueness = 0.55):
        """
        Plot orientation overlay using precomputed anisotropy.
        """

        theta_norm = theta / np.pi

        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap="gray")
        plt.imshow(
            theta_norm,
            cmap="hsv",
            alpha= opaqueness * anisotropy,
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

        counts, bin_edges = np.histogram(
            theta_flat,
            bins=bins,
            range=(0, np.pi),
            weights=weights_flat
        )

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        return counts, bin_centers, bin_width

# PLOTTING WEIGHTED HISTOGRAM
    @staticmethod
    def plot_orientation_histogram(counts, bin_centers, bin_width):
        """
        Plot weighted orientation histogram with HSV colouring.
        """

        colours = plt.cm.hsv(bin_centers / np.pi)

        plt.figure(figsize=(6, 6))

        plt.bar(
            bin_centers,
            counts,
            width=bin_width,
            color=colours,
            align='center'
        )

        plt.xlabel("Angle")
        plt.ylabel("Weighted pixel count")
        plt.title("Histogram of Dominant Orientations")

        plt.xlim(0, np.pi)

        plt.xticks(
            [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
            ["0", "π/4", "π/2", "3π/4", "π"]
        )

        plt.show()

# PLOTTING POLAR HISTOGRAM
    @staticmethod
    def plot_polar_histogram(counts, bin_centers, bin_width):
        """
        Plot weighted polar histogram.

        The histogram is duplicated over [pi, 2pi] because orientations are axial:
        angles differing by pi represent the same orientation.
        """

        angles_full = np.concatenate([bin_centers, bin_centers + np.pi])
        counts_full = np.concatenate([counts, counts])

        colours = plt.cm.hsv(bin_centers / np.pi)
        colours_full = np.concatenate([colours, colours], axis=0)

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
            diff = ImageAnalyzer.axial_distance(theta, center)
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
        tolerance_deg=10,
        opaqueness = 0.55
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
            alpha= opaqueness * anisotropy_energy,
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


        # Build alignment mask
        alignment = np.zeros_like(theta_flat, dtype=bool)

        for a in dominant_rads:
            alignment |= (ImageAnalyzer.axial_distance(theta_flat, a) < tolerance)

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

#AXIAL DISTANCE
    @staticmethod
    def axial_distance(a, b):
            """
            Axial distance (orientation invariant: π-periodic)
            In radians.

            Parameters:
            - a: angle1
            - b: angle2

            Returns:
            Axial distance between a and b.
            """
            d = np.abs(a - b) % np.pi
            return np.minimum(d, np.pi - d)


# -------------------
#COMPOSITE FUNCTIONS
#--------------------

#COMPUTING STRUCTURE TENSOR
    @staticmethod
    def compute_structure_tensor(img, sigma_derivative, sigma_tensor):
        """
        Computing the structure tensor for a given image

        Parameters:
        - img: grayscale image
        - sigma_derivative: sigma for first gaussian smoothing when finding gradients
        - sigma_tensor: sigma for second gaussan when smoothing the tensor over local neighbourhood pixels


        Returns:
        Structure tensor for given image
        """
        Ix, Iy = ImageAnalyzer.compute_gradients(img, sigma_deriv=sigma_derivative)
        S11, S12, S22 = ImageAnalyzer.structure_tensor(Ix, Iy)
        S11, S12, S22 = ImageAnalyzer.smooth_tensor(S11, S12, S22, sigma_tensor=sigma_tensor)
        S = ImageAnalyzer.tensor_to_matrix(S11, S12, S22)

        return S
    
#COMPUTING ANISOTROPY MASKED:
    @staticmethod
    def compute_anisotropy_masked(structure_tensor, percent = 45):
        """
        Computing the anisotropy masked for an image using the structure tensors

        Parameters:
        - structure_tensor: matrix containing structure tensors for the image
        - percent: the percentile energy cutoff, for when the anisotropy is included
        """


        evals, evecs = ImageAnalyzer.eigendecomposition(structure_tensor)
        lam_s, lam_l, v_s, v_l = ImageAnalyzer.split_eigenpairs(evals, evecs)
        energy = ImageAnalyzer.compute_energy(lam_s, lam_l)
        aniso = ImageAnalyzer.compute_isotropy_anisotropy(lam_s, lam_l)
        mask = ImageAnalyzer.energy_mask(energy, percent)
        anisotropy_masked = ImageAnalyzer.mask_anisotropy(aniso, mask)

        return anisotropy_masked
    

#-------------------
#PLOTTING FUNCTIONS
#-------------------

#PLOTTING ORIENTATION VECTORS
    @staticmethod
    def plot_orientation_vectors(img, structure_tensor, step = 40 ,scale = 50):
        """
        Plotting dominant orientation vectors

        Parameters:
        - img: grayscale image
        - structure_tenstor: matrix containing structure tensors for the image
        - step: How often vectors are displayed (pixel-wise)
        - scale: size of vectors. Smaller is bigger
        """

        evals, evecs = ImageAnalyzer.eigendecomposition(structure_tensor)
        _, _,v_small, _ = ImageAnalyzer.split_eigenpairs(evals,evecs)

        theta = ImageAnalyzer.orientation_from_eigenvectors(v_small)

        vx, vy = ImageAnalyzer.orientation_to_unit_vectors(theta)

        Y, X = np.mgrid[0:vx.shape[0], 0:vx.shape[1]]
        step = step

        # We are using the step so we only take every fourth
        Xs = X[::step, ::step]
        Ys = Y[::step, ::step]

        # We are only taking every fourth unit vector as well
        vxs = vx[::step, ::step]
        vys = vy[::step, ::step]

        plt.figure(figsize=(8,8))

        plt.imshow(img, cmap="gray")

        # plotting the dominant direction eigenvectors.
        plt.quiver(Xs, Ys, vxs, vys, color="yellow",scale = scale)
        plt.quiver(Xs, Ys, -vxs, -vys, color="yellow")

        plt.title("Dominant Orientation Vector Field")
        plt.show()

        

