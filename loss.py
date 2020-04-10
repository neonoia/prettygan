import numpy as np
import torch

criterion_l1 = torch.nn.L1Loss()

def criterion_histogram(src, ref, src_mask, ref_mask):
    src_mask = src_mask > 0
    ref_mask = ref_mask > 0
    src_masked = torch.masked_select(src, src_mask)
    ref_masked = torch.masked_select(ref, ref_mask)

    source = src_masked.numpy()
    template = ref_masked.numpy()
    
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    np_matched = interp_t_values[bin_idx].reshape(oldshape)
    src_matched = torch.from_numpy(np_matched)
    src_matched = src_matched.float()
    loss = criterion_l1(src_masked, src_matched)
    return loss