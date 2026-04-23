The codes implements the Lipschitz-inspired anisotropic diffusion equation using BDF2 semi-implicit Numerical Scheme
==================================================================================================================

    u_t = div( phi(|grad u|) grad u ) - lambda * (u - f)
    phi(s) = 1 / (1 + (s / K)^4)

Steps:
(1) First step: Backward Euler
(2) Subsequent steps: BDF2 semi-implicit numerical scheme
(3) Lagged diffusivity for the nonlinear diffusion coefficient
(4) Reference-based coarse-to-fine search for lambda and K.

Files included
---------------

Core solver:
-   denoisingBDF2.m

Adaptive parameter search:
-   denoisingBDF2_Optimal_lambda_K.m
-   save_parameter_search_history.m
-   save_lambda_search_history.m

Demo scripts:
-   demoRunDenoisingBDF2.m                % Demo script for running denoisingBDF with adaptive joint search for lambda and K
-   demoRunDenoisingBDF2_Fixed_lambda.m   % Demo script for running denoisingBDF with a fixed value of lambda

Utility functions:
-   load_grayscale_image.m
-   clip01.m
-   compute_psnr_basic.m
-   compute_ssim_basic.m

Datasets: % input noisy and clean images
-   data/01Noise.png
-   data/01.png
-   BSD68:  https://www.kaggle.com/code/mpwolke/berkeley-segmentation-dataset-68 
            Arbelaez, P., Maire, M., Fowlkes, C., & Malik, J. (2010). 
            Contour detection and hierarchical image segmentation. 
            IEEE transactions on pattern analysis and machine intelligence, 33(5), 898-916.
-   CBSD68: https://www.kaggle.com/code/mpwolke/berkeley-segmentation-dataset-68 
            Arbelaez, P., Maire, M., Fowlkes, C., & Malik, J. (2010). 
            Contour detection and hierarchical image segmentation. 
            IEEE transactions on pattern analysis and machine intelligence, 33(5), 898-916.
-   Set12:  https://www.kaggle.com/datasets/leweihua/set12-231008
            Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). 
            Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. 
            IEEE transactions on image processing, 26(7), 3142-3155.
-   Kodak24: https://r0k.us/graphics/kodak/ 
-   Urban100:   https://www.kaggle.com/datasets/harshraone/urban100
                Huang, J. B., Singh, A., & Ahuja, N. (2015). 
                Single image super-resolution from transformed self-exemplars. 
                In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5197-5206).
-   McMaster:   https://www.kaggle.com/datasets/sherylmehta/mcmaster 
                Zhang, L., Wu, X., Buades, A., & Li, X. (2011). 
                Color demosaicking by local directional interpolation and nonlocal adaptive thresholding. 
                Journal of Electronic imaging, 20(2), 023016-023016.

Output folder: % Folder collecting results of denoising
-   results/

Steps for running the codes
---------------------------

(1) Extract the distribution (ZIPPED) file to a preferred computer location
(2) Open MATLAB (the codes were prepared using MATLAB R2022a, but may be compatible with other versions of MATLAB
(3) Change the MATLAB working directory to the extracted package directory.
(4) Run: demoRunDenoisingBDF2

The Demo script will perform the following:
- Load the noisy and clean demo images
- Search jointly for the optimal values of lambda and K
- Denoise the image with the selected pair
- Save the denoised image and search history in results/

Expected outputs
----------------

The Demo script saves the following elements:
- results/denoisingBDF2_Optimal_lambda_K.png
- results/adaptive_lambda_K_history.csv
- results/adaptive_lambda_K_summary.txt

Additional information
----------------------

(1) Joint optimization of PSNR and SSIM requires a clean reference image.
    Hence, the adaptive lambda-K search is intended for benchmarking,
    calibration, and controlled experiments.

(2) The selected pair is the best pair found among the evaluated candidates.
    In 'balanced' mode (eqn. (2.18) in the paper), the search chooses the point closest to the ideal
    normalized PSNR-SSIM point. Individual values of 'psnr' or 'ssim' may as well be selected for optimization

(3) The code does not require any special toolbok for PSNR and SSIM evaluation.

4) The PDE solver uses PCG ( Preconditioned Conjugate Gradient) with a diagonal preconditioner to solve eqn. (2.5) in the paper. If PCG does not
   converge sufficiently, the code falls back to MATLAB's sparse backslash operation.

*** Paper: Lipschitz energy functional for diffusion-inspired denoising methods
*** Submitted to The Visual Computer for consideration 

Code customization
---------------------

The search ranges of lambda and K can be edited in the demoRunDenoisingBDF2.m

Example:
    searchOpts.lambdaCandidates = logspace(log10(0.005), log10(1.2), 9);
    searchOpts.KCandidates = logspace(log10(0.01), log10(0.30), 9);
    searchOpts.refinePasses = 3;
    searchOpts.refineLambdaPoints = 5;
    searchOpts.refineKPoints = 5;
    searchOpts.selectionCriterion = 'balanced';   % or 'psnr', 'ssim'

