import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sci 

class ImageAnalyzer: 

# LOADING GRAYSCALE
    @staticmethod
    def load_crop_grayscale(path, crop=None):
        """
        Expected to be used on either Grayscale- or RGB images.
        Load image, optionally crop, and convert to grayscale.

        Parameters
        ----------------------
            path:
                Relative path to image file

            crop:
                (row_start, row_end, col_start, col_end)

        Returns:
            grayscale image:
        """
        img = plt.imread(path)
        if crop is not None:
            r1, r2, c1, c2 = crop
            img = img[r1:r2, c1:c2]
        if img.ndim == 3: 
            img = img.mean(axis=2)
        return img


# ORIENTATION HISTOGRAM
    @staticmethod
    def compute_orientation_histogram(theta, anisotropy_energy = None, bins=180):
        """
        Compute histogram of orientations.
        Can be unweighted or weighted

        Parameters
        ----------------------
            theta:
                Orientation field (radians, [0, pi])

            anisotropy_energy:
                Weights (same shape as theta)

            bins:
                Number of histogram bins (defaults to 180)

        Returns:
            counts, bin_centers, bin_width:
        """

        theta_flat = theta.ravel()
        
        if anisotropy_energy is None:
            weights_flat = np.ones_like(theta_flat)
        else:
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


# ORIENTATION BINNING
    @staticmethod
    def bin_orientations(theta, centers, tolerance_deg=10):
        """
        Assign each pixel orientation to nearest dominant orientation (if within tolerance).

        Parameters
        ----------------------
            theta (np.ndarray):
                Orientation field in radians with values in the range [0, pi].

            centers (list[float] | np.ndarray):
                Dominant orientations in degrees.

            tolerance_deg (float, optional):
                Angular tolerance for assigning orientations to a bin, in degrees.
                Defaults to 10.

        Returns:
            chosen_theta_binned (np.ndarray):
                Same shape as theta, NaN where no match
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
    
#BINS ORIENTATIONS UNIFORMLY
    @staticmethod
    def bin_orientations_uniform(theta, bin_size):
        """
        Quantize orientations into fixed angular bins.

        Parameters
        ----------------------
            theta:
                Orientation field in radians

            bin_size:
                Bin spacing in degrees

        Returns:
            theta_binned: in radians
        """

        bin_size = np.deg2rad(bin_size)

        theta_binned = np.round(theta / bin_size) * bin_size
        theta_binned = np.mod(theta_binned, np.pi)

        return theta_binned

#COMPUTING ALIGNMENT PERCENTAGE
    @staticmethod
    def compute_alignment_percentage(theta, anisotropy_energy, dominant_angles_deg, tolerance_deg=10):
        """
        Compute percentage of weighted pixels aligned with dominant orientations.

        Parameters
        ----------------------
            theta:
                Orientations field in radians

            anisotropy_energy:
                Computed ansisotropy

            dominant_angles_deg:
                List of chosen dominant orientations in degrees

            tolerance_deg:
                Grouping tolerance in degrees (defaults to 10)

        Returns:
            percentage: (0–100)
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
    def print_alignment_percentage(percentage, dominant_angles_deg, tolerance_deg=10):
        """
        print alignment percentage.

        Parameters
        ----------------------
            percentage:
                Computed alignment percentage

            dominant_angles_deg:
                List of dominant angles in degrees

            tolerance_deg:
                Grouping tolerance in degrees
        """
        print(
            f"{percentage:.2f}% of orientations align with "
            f"{dominant_angles_deg}° (±{tolerance_deg}°)"
        )



#--------------------------
#HELPER/LOW LEVEL FUNCTIONS
#--------------------------

# GRADIENTS
    @staticmethod
    def compute_gradients(img, sigma_deriv=1):
        """
        Compute image gradients using Gaussian derivatives.

        Returns:
            Ix, Iy:
        """
        Ix = sci.gaussian_filter(img, sigma=sigma_deriv, order=[0, 1])
        Iy = sci.gaussian_filter(img, sigma=sigma_deriv, order=[1, 0])
        return Ix, Iy

#CREATING STRUCTURE TENSOR
    @staticmethod
    def structure_tensor(Ix, Iy):
        """
        Compute raw structure tensor components.
        Returns:
            S11, S12, S22:
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

        Returns:
            S11, S12, S22:
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

        Returns:
            S:
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

        Returns
        ----------------------------
            eigen values: 
                (H, W, 2)

            eigen vectors: 
                (H, W, 2, 2)

        """
        evals, evecs = np.linalg.eigh(S)
        return evals, evecs

#FIND LAMBDA SMALL AND LARGE 
    @staticmethod
    def split_eigenpairs(evals, evecs):
        """
        Split eigenvalues and eigenvectors into small and large components.

        Returns:
            lam_small, lam_large, v_small, v_large:

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

        Returns:
            Theta:

        """
        theta = np.arctan2(-v_small[:, :, 1], v_small[:, :, 0])
        theta = np.mod(theta, np.pi)
        return theta

#CREATE UNIT VECTOR FOR DOMINANT ORIENTATION
    @staticmethod
    def orientation_to_unit_vectors(theta):
        """
        Convert orientation angle to unit direction vectors.

        Returns:
            vx, vy:
                x and y components of the unit vectors

        """
        vx = np.cos(theta)
        vy = np.sin(theta)
        return vx, vy
    
#COMPUTE TOTAL ENERGY PER PIXEL 
    @staticmethod
    def compute_energy(lam_small, lam_large):
        """
        Compute structure tensor energy.

        Returns:
            energy:

        """
        return lam_small + lam_large

#COMPUTE ISOTROPY AND ANISOTROPY
    @staticmethod
    def compute_anisotropy(lam_small, lam_large):
        """
        Compute isotropy and anisotropy measures.

        Returns:
            anisotropy:

        """
        isotropy = lam_small / (lam_large + 1e-10)
        anisotropy = 1 - isotropy
        return anisotropy
    
#COMPUTE ENERGY CUTOFF MASK
    @staticmethod
    def energy_mask(energy, percentile=45):
        """
        Create mask for high-energy pixels.

        Returns:
            mask:

        """
        threshold = np.percentile(energy, percentile)
        return energy >= threshold
    
#APPLY ENERGY CUTOFF TO ANISOTROPY
    @staticmethod
    def mask_anisotropy(anisotropy, mask):
        """
        Zero out anisotropy where energy is low.

        Returns:
            masked results:

        """
        result = anisotropy.copy()
        result[~mask] = 0
        return result
    
#AXIAL DISTANCE
    @staticmethod
    def axial_distance(a, b):
            """
            Axial distance (orientation invariant: π-periodic)
            In radians.

            Parameters
            ----------------------
                a:
                    Angle1

                b:
                    Angle2

            Returns:
                Axial distance between a and b:
            """
            d = np.abs(a - b) % np.pi
            return np.minimum(d, np.pi - d)


#--------------------
#COMPOSITE FUNCTIONS
#--------------------

#COMPUTING STRUCTURE TENSOR
    @staticmethod
    def compute_structure_tensor(img, sigma_derivative, sigma_tensor):
        """
        Computing the structure tensor for a given image

        Parameters
        ----------------------
            img:
                Grayscale image

            sigma_derivative:
                Sigma for first gaussian smoothing when finding gradients

            sigma_tensor:
                Sigma for second gaussian smoothing when smoothing the tensor over local neighbourhood pixels

        Returns:
            Smoothed Structure tensor for given image:
        """
        Ix, Iy = ImageAnalyzer.compute_gradients(img, sigma_deriv=sigma_derivative)
        S11, S12, S22 = ImageAnalyzer.structure_tensor(Ix, Iy)
        S11, S12, S22 = ImageAnalyzer.smooth_tensor(S11, S12, S22, sigma_tensor=sigma_tensor)
        S = ImageAnalyzer.tensor_to_matrix(S11, S12, S22)

        return S
    
#COMPUTING ANISOTROPY MASKED:
    @staticmethod
    def compute_anisotropy_masked(structure_tensor, percentile = 45):
        """
        Computing the anisotropy masked for an image using the structure tensors

        Parameters
        ----------------------
            structure_tensor:
                Matrix containing structure tensors for the image

            percentile:
                The percentile energy cutoff, for when the anisotropy is included

        Returns:
            The anisotropy masked with percentile value energy cutoff:
        """


        evals, evecs = ImageAnalyzer.eigendecomposition(structure_tensor)
        lam_s, lam_l, v_s, v_l = ImageAnalyzer.split_eigenpairs(evals, evecs)
        aniso = ImageAnalyzer.compute_anisotropy(lam_s, lam_l)
        energy = ImageAnalyzer.compute_energy(lam_s, lam_l)
        mask = ImageAnalyzer.energy_mask(energy, percentile)
        anisotropy_masked = ImageAnalyzer.mask_anisotropy(aniso, mask)

        return anisotropy_masked
    

#COMPUTING UNIT VECTOR OF STRUCTURE TENSOR FOLLOWING ORIENTATION
    @staticmethod
    def compute_orientation_unit_vector(structure_tensor):
        """
        Computes the orientation vectors for the smallest eigenvalues

        Parameters
        ----------------------
            structure_tensor:
                The matrix of structure tensors for the image

        Returns:
            vx, vy:
                x and y components of the unit vectors
        """
    
        evals, evecs = ImageAnalyzer.eigendecomposition(structure_tensor)
        _, _,v_small, _ = ImageAnalyzer.split_eigenpairs(evals,evecs)
        theta = ImageAnalyzer.orientation_from_eigenvectors(v_small)
        vx, vy = ImageAnalyzer.orientation_to_unit_vectors(theta)

        return vx,vy


#-------------------
#PLOTTING FUNCTIONS
#-------------------

#PLOT IMAGE
    @staticmethod
    def plot_image(img, title = "Grayscale Image" ):
        """Display image in grayscale."""
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.show()

#PLOTTING ORIENTATION VECTORS
    @staticmethod
    def plot_orientation_vectors(img, structure_tensor, step = 40 ,scale = 50):
        """
        Plotting dominant orientation vectors

        Parameters
        ----------------------
            img:
                Grayscale image

            structure_tenstor:
                Matrix containing structure tensors for the image

            step:
                How often vectors are displayed pixel-wise (defaults to 40)

            scale:
                Size of vectors. Smaller is bigger (defaults to 50)
        """

        vx, vy = ImageAnalyzer.compute_oriention_unit_vector(structure_tensor)

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
    
#PLOT ORIENTATION OVERLAY WITH CHOSEN ENERGY CUTOFF
    @staticmethod
    def plot_orientation_overlay(image, theta, anisotropy = 1, percentile = 0, opaqueness = 0.55, title: str | None = None):
        """
        Plot orientation overlay using precomputed anisotropy.

        Parameters
        ----------------------
            image:
                The grayscale image

            theta:
                Orientation field in radians

            anisotropy:
                List of anisotropy value for given image, same shape as theta

            percentile:
                The percentile energy cutoff, for when the anisotropy is computed

            opaqueness:
                Opaqueness of color when plotted
        """

        theta_norm = theta / np.pi

        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap="gray")
        plt.imshow(
            theta_norm,
            cmap="hsv",
            alpha= opaqueness * anisotropy,
            interpolation="nearest",
            vmin=0,
            vmax=1

        )
        if not title:
            plt.title(f"Top {100 - percentile}% of Energy Pixels")
        else:
            plt.title(title)
        plt.show()

        

# PLOTTING WEIGHTED HISTOGRAM
    @staticmethod
    def plot_orientation_histogram(counts, bin_centers, bin_width, weighted: bool):
        """
        Plot orientation histogram with HSV colouring.

        Can either be weighted or not.

        Parameters
        ----------------------
            counts:
                Counts of each bin from the compute_orientation_histogram

            bin_centers:
                Centers of each bin from the compute_orientation_histogram

            bin_width:
                Width of each bin from the compute_orientation_histogram. Can also be chosen arbitrarily

            weigthed:
                Bool. If true displays weighted text. If false display unweighted figure text
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

        if weighted:
            plt.ylabel("Weighted pixel count")
            plt.title("Weighted Histogram of Dominant Orientations")

        else:
            plt.ylabel("Pixel count")
            plt.title("Histogram of Dominant Orientations")

        

        plt.xlim(0, np.pi)

        plt.xticks(
            [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
            ["0", "π/4", "π/2", "3π/4", "π"]
        )

        plt.show()

# PLOTTING POLAR HISTOGRAM
    @staticmethod
    def plot_polar_histogram(counts, bin_centers, bin_width, weighted: bool):
        """
        Plot polar histogram.
        can be weighted or unweighted.

        The histogram is duplicated over [pi, 2pi] because orientations are axial:
        angles differing by pi represent the same orientation.

        Parameters
        ----------------------
            counts:
                Counts of each bin from the compute_orientation_histogram

            bin_centers:
                Centers of each bin from the compute_orientation_histogram

            bin_width:
                Width of each bin from the compute_orientation_histogram. Can also be chosen arbitrarily

            weigthed:
                Bool. If true displays weighted text. If false display unweighted figure text
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
        if weighted:
            plt.title("Weighted Polar Histogram")   
        else:
            plt.title("Polar histogram")
        plt.show()