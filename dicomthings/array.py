import numpy as np
from scipy.ndimage import zoom, gaussian_filter, uniform_filter, distance_transform_edt


def downsample_array(input_array, window_sizes):
    """Downsamples an input array using a sliding window and averaging over the window.

    This function performs a sliding window with a stride equal to the window size over each dimension of the input array with the specified window size, and
    then averages over the window to produce a single output value. If the size of the input array is not divisible by the
    window size, the output array will be truncated to the largest integer size that is divisible by the window size.

    Parameters
    ----------
    input_array : numpy.ndarray
        Input array to be downsampled.
    window_sizes : tuple or list of int
        Window size for each dimension of the input array. If only one value is specified, the same window size is used
        for all dimensions.

    Returns
    -------
    output_array : numpy.ndarray
        Downsampled array.

    Raises
    ------
    AssertionError
        If `window_sizes` is not a tuple or list, or if not all dimensions in the input array have a corresponding window
        size being specified, or if any of the window sizes are not >= 1 integers.
    """

    if not isinstance(window_sizes, (tuple, list)):
        window_sizes = [window_sizes] * len(input_array.shape)

    assert len(input_array.shape) == len(window_sizes), "Not all dimensions in the input array have a corresponding window size being specified."
    assert all([isinstance(window_size, int) and window_size >= 1 for window_size in window_sizes]), "Window sizes must be >= 1 integers."
    output_shapes = []
    tmp_shapes = []
    for window_size, input_shape in zip(window_sizes, input_array.shape):
        output_shape = input_shape if window_size == 1 else int(input_shape - input_shape % window_size)
        output_shapes.append(output_shape)
        tmp_shapes += [output_shape // window_size, window_size]

    tmp_array = np.reshape(input_array[tuple([slice(output_shape) for output_shape in output_shapes])], tmp_shapes)
    output_array = np.mean(tmp_array, tuple(range(1, len(tmp_shapes), 2)))
    return output_array


def crop(I, S_size, coordinates=None, subsample_factors=None, default_value=0, prefilter=None):
    """Crop an input array I to size S_size centered on coordinates in I with optional subsampling.

    Parameters
    ----------
    I : np.ndarray
        Input array. Must be at least as large as S_size in each dimension.
    S_size : tuple, list
        Output array size of the cropped volume.
    coordinates : tuple, list or None, optional
        Indices in I to center the crop. If None, the center of I is used.
    subsample_factors : tuple, list of int or None, optional
        Subsampling factors for each requested segment dimension.
    default_value : int or np.nan, optional
        Value to fill outside of I. If np.nan, a nearest-neighbor interpolation is used to fill the edges.
    prefilter : {'gaussian', 'uniform'} or None, optional
        Filter to use during interpolation. Uniform filter uses the `downsample_array` function.

    Returns
    -------
    np.ndarray
        Cropped array.
    """

    assert I.ndim >= len(S_size), "Number of image dimensions must be equal to or larger than the number of requested segment sizes. Extra (trailing) dimensions will not be cropped."
    if coordinates is None:
        coordinates = [s // 2 for s in I.shape]  # cropping happens around the center

    else:
        assert len(coordinates) == len(S_size), "Coordinates must specify an index for each requested segment dimension."
        coordinates = [s // 2 if c is None else c for c, s in zip(coordinates, I.shape[:len(coordinates)])]  # cropping happens around the specified coordinates
        coordinates += [s // 2 for s in I.shape[len(coordinates):]]  # cropping happens around the center of the remaining trailing dimensions

    if subsample_factors is None:
        subsample_factors = [1] * len(I.shape)  # no resampling happens

    else:
        assert len(subsample_factors) == len(S_size), "A subsample factor must be specified for each requested segment dimension."
        subsample_factors = list(subsample_factors) + [1] * (I.ndim - len(subsample_factors))

    if prefilter is not None:
        assert prefilter in ["uniform", "gaussian"]
        if prefilter == "gaussian":
            I = gaussian_filter(I.astype(np.float32), [s_f if s_f > 1 else 0 for s_f in subsample_factors], mode="nearest")

        elif prefilter == "uniform":
            I = uniform_filter(I.astype(np.float32), subsample_factors, mode="nearest")

    S_size = tuple(S_size) + tuple(I.shape[len(S_size):])
    S = np.full(S_size, fill_value=default_value, dtype=np.float32)
    idx_I = [slice(None)] * I.ndim
    idx_S = [slice(None)] * S.ndim
    for i, (d_I, d_S, c, s_f) in enumerate(zip(I.shape, S_size, coordinates, subsample_factors)):
        n_left_I = c
        n_right_I = d_I - c - 1
        n_left_S = d_S // 2
        n_right_S = d_S // 2
        if d_S % 2 == 0:
            n_right_S -= 1

        if n_left_I < n_left_S * s_f:
            n = n_left_I // s_f
            start_S = d_S // 2 - n
            start_I = c - n * s_f

        else:
            start_S = 0
            start_I = c - n_left_S * s_f

        if n_right_I < n_right_S * s_f:
            n = n_right_I // s_f
            end_S = d_S // 2 + n
            end_I = c + n * s_f

        else:
            end_S = d_S - 1
            end_I = c + n_right_S * s_f

        idx_I[i] = slice(start_I, end_I + 1, s_f)
        idx_S[i] = slice(start_S, end_S + 1)

    S[tuple(idx_S)] = I[tuple(idx_I)]
    if np.any(np.isnan(default_value)):
        S = S[tuple(distance_transform_edt(np.isnan(S), return_distances=False, return_indices=True))]

    return S


def put(I, S, coordinates=None, subsample_factors=None):
    """This function can be used to put back a segment centered on a specific coordinate in the input array (reverse of crop function).

    Parameters
    ----------
    I : np.ndarray
        The input array in which to put back the segment S. Can be of any number of dimensions.
    S : np.ndarray
        The segment array to put back in I. Same number of dimensions as I.
    coordinates : None or tuple or list
        Around what coordinate in the input array the segment array will be put back. When None, the center coordinate in I will be used.
        None can also be used for a specific axis to specify that the center coordinate along that axis must be used.
    subsample_factors : None or tuple or list
        If the segment array was subsampled, one can specify an integer subsample factor for each axis.
        None is used to denote a subsample factor of 1.

    Returns
    -------
    I : np.ndarray
        The output array I in which the segment S is now put.
    """

    assert I.ndim == S.ndim, "Number of image dimensions must be equal to number of segment dimensions."
    if coordinates is None:
        coordinates = [s // 2 for s in I.shape]  # putting happens around the center

    else:
        assert len(coordinates) <= I.ndim
        coordinates = [s // 2 if c is None else c for c, s in zip(coordinates, I.shape[:len(coordinates)])]  # putting happens around the specified coordinates
        coordinates += [s // 2 for s in I.shape[len(coordinates):]]  # putting happens around the center of the remaining trailing dimensions

    if subsample_factors is None:
        subsample_factors = [1] * len(I.shape)

    else:
        assert len(subsample_factors) == I.ndim, "A subsample factor must be specified for each image/segment dimension."

    idx_I = [slice(None)] * I.ndim
    idx_S = [slice(None)] * S.ndim
    for i, (d_I, d_S, c, s_f) in enumerate(zip(I.shape, S.shape, coordinates, subsample_factors)):
        n_left_I = c
        n_right_I = d_I - c - 1
        n_left_S = d_S // 2
        n_right_S = d_S // 2
        if d_S % 2 == 0:
            n_right_S -= 1

        if n_left_I < n_left_S * s_f:
            n = n_left_I // s_f
            start_S = d_S // 2 - n
            start_I = c - n * s_f

        else:
            start_S = 0
            start_I = c - n_left_S * s_f

        if n_right_I < n_right_S * s_f:
            n = n_right_I // s_f
            end_S = d_S // 2 + n
            end_I = c + n * s_f

        else:
            end_S = d_S - 1
            end_I = c + n_right_S * s_f

        idx_I[i] = slice(start_I, end_I + 1, s_f)
        idx_S[i] = slice(start_S, end_S + 1)

    I[tuple(idx_I)] = S[tuple(idx_S)]
    return I
