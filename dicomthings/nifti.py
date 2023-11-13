import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter


def reorient_nifti(image, output_orientation="LPS"):
    """Reorients a NIfTI image to the specified output orientation.

    Parameters:
    -----------
    image : nibabel.Nifti1Image
        The input NIfTI image to be reoriented.
    output_orientation : str, optional
        The desired output orientation code in the form of "LPS", "RAS", etc.
        Default is "LPS".

    Returns:
    --------
    reoriented_image : nibabel.Nifti1Image
        The reoriented 3D NIfTI image.
    """

    ornt = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(image.affine))
    ornt_ = nib.orientations.axcodes2ornt(output_orientation)
    return image.as_reoriented(nib.orientations.ornt_transform(ornt, ornt_))


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


def resample_nifti(input_nii, output_zooms, order=3, prefilter=True, reference_nii=None):
    """Resamples a Nifti image to have new voxel sizes defined by `output_zooms`.

    Parameters
    ----------
    input_nii : nibabel.nifti1.Nifti1Image
        The input Nifti image to resample.
    output_zooms : tuple
        The desired voxel size for the resampled Nifti image. It must be a tuple or list of same length as the number of dimensions in the original Nifti image. None can be used to specify that a certain dimension does not need to be resampled.
    order : int or str, optional
        The order of interpolation (0 to 5) or the string 'mean' to perform a downsampling based on mean (with a moving average window that gets the effective output zooms as close as possible to the requested output zooms). Default is 3.
    prefilter : bool, optional
        Whether to apply a Gaussian filter to the input data before resampling (https://stackoverflow.com/questions/35340197/box-filter-size-in-relation-to-gaussian-filter-sigma). Default is True.
    reference_nii : nibabel.nifti1.Nifti1Image or None, optional
        If not None, the resampled Nifti image will have the same dimensions as `reference_nii`. Default is None.
        E.g., when doing upsampling back to an original image space we want to make sure the image sizes are consistent. By giving a reference Nifti image, we crop or pad with zeros where necessary.

    Returns
    -------
    output_nii : nibabel.nifti1.Nifti1Image
        The resampled Nifti image.
    """

    input_zooms = input_nii.header.get_zooms()
    assert len(input_zooms) == len(output_zooms), "Number of dimensions mismatch."
    assert np.allclose(input_zooms[:3], np.linalg.norm(input_nii.affine[:3, :3], 2, axis=0)), "Inconsistency (we only support voxel size = voxel distance) in affine and zooms (spatial) of input Nifti image."
    input_array = input_nii.get_fdata()
    output_zooms = [input_zoom if output_zoom is None else output_zoom for input_zoom, output_zoom in zip(input_zooms, output_zooms)]
    if order == "mean":
        assert all([4 / 3 * output_zoom >= input_zoom for output_zoom, input_zoom in zip(output_zooms, input_zooms)]), "This function with order='mean' only supports downsampling by an integer factor (input zooms: {}).".format(input_zooms)
        zoom_factors = [1 if input_zoom > 2 / 3 * output_zoom else 1 / int(round(output_zoom / input_zoom)) for input_zoom, output_zoom in zip(input_zooms, output_zooms)]
        output_zooms = [input_zoom / zoom_factor for input_zoom, zoom_factor in zip(input_zooms, zoom_factors)]
        output_array = downsample_array(input_array, window_sizes=[int(1 / zoom_factor) for zoom_factor in zoom_factors])

    else:
        assert isinstance(order, int), "When order != 'mean', it must be an integer (see scipy.ndimage.zoom)."
        zoom_factors = [input_zoom / output_zoom for input_zoom, output_zoom in zip(input_zooms, output_zooms)]
        if prefilter:
            input_array = gaussian_filter(input_array.astype(np.float32), [np.sqrt(((1 / zoom_factor)**2 - 1) / 12) if zoom_factor < 1 else 0 for zoom_factor in zoom_factors], mode="nearest")

        output_array = zoom(input_array, zoom_factors, order=order, mode="nearest")

    if reference_nii is not None:
        assert np.allclose(output_zooms, reference_nii.header.get_zooms()), "The output zooms are not equal to the reference zooms."
        output_array_ = np.zeros_like(reference_nii.get_fdata())
        output_array_[tuple([slice(min(s, s_)) for s, s_ in zip(output_array.shape, output_array_.shape)])] = output_array[tuple([slice(min(s, s_)) for s, s_ in zip(output_array.shape, output_array_.shape)])]
        output_array = output_array_
        output_affine = reference_nii.affine

    else:
        output_affine = input_nii.affine.copy()
        output_affine[:3, :3] = output_affine[:3, :3] / zoom_factors[:3]

    output_nii = nib.Nifti1Image(output_array, affine=output_affine)
    output_nii.header.set_zooms(output_zooms)  # important to set non-spatial zooms correctly
    return output_nii
