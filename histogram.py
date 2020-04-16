import numpy as np
import torch
from torch.utils.cpp_extension import load

cpp = torch.utils.cpp_extension.load(name="histogram_cpp", sources=["histogram.cpp", "histogram.cu"])

def mask_regions(src, ref, src_mask, ref_mask):
    src = src.data.clone()
    ref = ref.data.clone()

    channels_A = list(torch.split(src, 1, 1))
    channels_B = list(torch.split(ref, 1, 1))

    src_mask = src_mask > 0
    ref_mask = ref_mask > 0

    src_masked_1 = torch.masked_select(channels_A[0], src_mask)
    src_masked_2 = torch.masked_select(channels_A[1], src_mask)
    src_masked_3 = torch.masked_select(channels_A[2], src_mask)
    src_masked = torch.cat([src_masked_1.unsqueeze(0), src_masked_2.unsqueeze(0), src_masked_3.unsqueeze(0)], 0)

    ref_masked_1 = torch.masked_select(channels_B[0], ref_mask)
    ref_masked_2 = torch.masked_select(channels_B[1], ref_mask)
    ref_masked_3 = torch.masked_select(channels_B[2], ref_mask)
    ref_masked = torch.cat([ref_masked_1.unsqueeze(0), ref_masked_2.unsqueeze(0), ref_masked_3.unsqueeze(0)], 0)

    return src_masked, ref_masked

# ----------------------
#  Numpy Histogram Crit
# ----------------------

def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = diff <= -1
    # We need to mask the negative differences
    # since we are looking for values above
    if torch.all(mask):
        c = torch.abs(diff).argmin()
        return c # returns min index of the nearest if target is greater than any value
    mask = mask < 1
    masked_diff = torch.masked_select(diff, mask)
    return masked_diff.argmin()

def histogram_matching(src, ref, src_mask, ref_mask):
    src_mask = src_mask > 0
    ref_mask = ref_mask > 0
    source = torch.masked_select(src, src_mask)     # masked source
    template = torch.masked_select(ref, ref_mask)   # masked template
    
    oldshape = source.shape

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = torch.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = torch.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = torch.cumsum(s_counts, dim=0).float()
    sc = s_quantiles.clone()
    s_quantiles /= sc[-1]
    t_quantiles = torch.cumsum(t_counts, dim=0).float()
    tc = t_quantiles.clone()
    t_quantiles /= tc[-1]

    # Round the values
    sour = (s_quantiles*255).round()
    temp = (t_quantiles*255).round()

    # Map the rounded values
    b = []
    for data in sour[:]:
        b.append(find_nearest_above(temp, data))
    b = torch.tensor(b)

    src_matched = b[bin_idx].reshape(oldshape).cuda()
    src_matched = src_matched.float()
    return source, src_matched

# ----------------------
#  Cuda Histogram Crit
# ----------------------

def get_min_max(input):
    return torch.min(input[0].view(input.shape[1], -1), 1)[0].data.clone(), \
        torch.max(input[0].view(input.shape[1], -1), 1)[0].data.clone()
    
def calc_hist(input, target, min_val, max_val):
    res = input.data.clone() 
    cpp.matchHistogram(res, target.clone())
    for c in range(res.size(0)):
        res[c].mul_(max_val[c] - min_val[c]) 
        res[c].add_(min_val[c])                      
    return res.data

def histogram_matching_cuda(src, ref):
    target_min, target_max = get_min_max(ref)
    target_hist = cpp.computeHistogram(src, 256)
    src_matched = calc_hist(src, target_hist, target_min, target_max)
    return src_matched