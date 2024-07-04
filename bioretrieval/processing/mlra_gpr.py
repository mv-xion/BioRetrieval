import numpy as np
import math
from multiprocessing import Pool, cpu_count


# GPR function
def GPR_mapping_pixel(args):
    pixel_spectra, hyp_ell_GREEN, X_train_GREEN, mean_model_GREEN, hyp_sig_GREEN, \
        XDX_pre_calc_GREEN, alpha_coefficients_GREEN, Linv_pre_calc_GREEN, hyp_sig_unc_GREEN = args
    """
    GPR retrieval function for one pixel:
    input arguments are given in the main parallel function
    outputs the mean and variance value calculated
    """
    # Image is already normalised
    im_norm_ell2D = pixel_spectra

    # Multiply with hyperparams
    im_norm_ell2D_hypell = im_norm_ell2D * hyp_ell_GREEN

    # flatten toarray toarray1
    im_norm_ell2D = im_norm_ell2D.reshape(-1, 1)
    im_norm_ell2D_hypell = im_norm_ell2D_hypell.reshape(-1, 1)

    PtTPt = np.matmul(np.transpose(im_norm_ell2D_hypell), im_norm_ell2D).ravel() * (-0.5)
    PtTDX = np.matmul(X_train_GREEN, im_norm_ell2D_hypell).ravel().flatten()

    arg1 = np.exp(PtTPt) * hyp_sig_GREEN
    k_star = np.exp(PtTDX - (XDX_pre_calc_GREEN.ravel() * 0.5))

    mean_pred = (np.dot(k_star.ravel(), alpha_coefficients_GREEN.ravel()) * arg1) + mean_model_GREEN
    filterDown = np.greater(mean_pred, 0).astype(int)
    mean_pred = mean_pred * filterDown

    k_star_uncert = np.exp(PtTDX - (XDX_pre_calc_GREEN.ravel() * 0.5)) * arg1
    Vvector = np.matmul(Linv_pre_calc_GREEN, k_star_uncert.reshape(-1, 1)).ravel()

    Variance = math.sqrt(abs(hyp_sig_unc_GREEN - np.dot(Vvector, Vvector)))

    return mean_pred.item(), Variance


def GPR_mapping_parallel(image, hyp_ell_GREEN, X_train_GREEN, mean_model_GREEN, hyp_sig_GREEN,
                         XDX_pre_calc_GREEN, alpha_coefficients_GREEN, Linv_pre_calc_GREEN, hyp_sig_unc_GREEN):
    """
    GPR function parallel processing:

    Input parameters: image(dim,y,x), hyperparams mx, X train, mean model,
    hyperparam estimate, XDX, alpha coefficients, Linverse, hyperparams uncertainty mx

    Output: retrieved variable map, map of uncertainty
    """
    ydim, xdim = image.shape[1:]

    variable_map = np.empty((ydim, xdim))
    uncertainty_map = np.empty((ydim, xdim))

    args_list = [(image[:, f, v], hyp_ell_GREEN, X_train_GREEN, mean_model_GREEN,
                  hyp_sig_GREEN, XDX_pre_calc_GREEN, alpha_coefficients_GREEN, Linv_pre_calc_GREEN, hyp_sig_unc_GREEN)
                 for f in range(ydim) for v in range(xdim)]

    with Pool(processes=cpu_count()) as pool:  # process on different cpus (as many as we have)
        results = pool.map(GPR_mapping_pixel, args_list)

    for i, (mean_pred, Variance) in enumerate(results):
        f = i // xdim
        v = i % xdim
        variable_map[f, v] = mean_pred
        uncertainty_map[f, v] = Variance

    return variable_map, uncertainty_map
