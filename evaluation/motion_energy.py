import glob
import os

import cv2
import matplotlib.pyplot as plt
import moten
import numpy as np
from PIL import Image
from scipy import misc

def rgb2lab(image):
    '''Convert RGB to CIE LAB color space **in-place**

    Parameters
    ----------
    image : 3D numpy float array
        Array must be in [0,1] range. Last
        dimension corresponds to RGB channels.

    Returns
    -------
    LAB : 3D numpy float array
        The CIE LAB representation in the image.
    '''

    WhitePoint = np.asarray([0.95047,1.0,1.08883])
    # Convert image to XYZ
    image = rgb2xyz(image)
    # % Convert XYZ to CIE L*a*b*
    X = image[:,:,0]/WhitePoint[0]
    Y = image[:,:,1]/WhitePoint[1]
    Z = image[:,:,2]/WhitePoint[2]
    fX = _ff(X)
    fY = _ff(Y)
    fZ = _ff(Z)
    image[:,:,0] = 116.0*fY - 16          # L*
    image[:,:,1] = 500.0*(fX - fY)        # a*
    image[:,:,2] = 200.0*(fY - fZ)        # b*
    return image

def imagearray2luminance(uint8arr, size=None, filter=Image.ANTIALIAS, dtype=np.float64):
    '''Convert an array of uint8 RGB images to a luminance image

    Parameters
    ----------
    uint8arr : 4D np.ndarray, (nimages, vdim, hdim, color)
        The uint8 RGB frames.
    size : optional, tuple (vdim, hdim)
        The desired output image size.
    filter: to be passed to PIL

    Returns
    -------
    luminance_array : 3D np.ndarray, (nimages, vdim, hdim)
        The luminance image representation.
        Pixel values are in the 0-100 range.
    '''
    if uint8arr.ndim == 3:
        # handle single image case
        uint8arr = np.asarray([uint8arr])

    nimages, vdim, hdim, cdim = uint8arr.shape
    outshape = (nimages, vdim, hdim) if size is None \
        else (nimages, size[0], size[1])

    luminance = np.zeros(outshape, dtype=dtype)
    for imdx in range(nimages):
        im = uint8arr[imdx]
        if size is not None:
            im = Image.fromarray(im)
            im = resize_image(im, size=size, filter=filter)
        im = rgb2lab(im/255.)[...,0]
        luminance[imdx] = im
    return luminance

def rgb2xyz(image):
    '''Convert RGB to CIE XYZ color space **in-place**

    Parameters
    ----------
    image : 3D numpy float array
        Array must be in [0,1] range. Last
        dimension corresponds to RGB channels.

    Returns
    -------
    LAB : 3D numpy float array
        The CIE XYZ representation in the image.
    '''
    # % Undo gamma correction
    R = _invgammacorrection(image[:,:,0])
    G = _invgammacorrection(image[:,:,1])
    B = _invgammacorrection(image[:,:,2])
    # % Convert RGB to XYZ
    xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                            [0.212671, 0.715160, 0.072169],
                            [0.019334, 0.119193, 0.950227]])
    T = xyz_from_rgb.T.ravel()
    image[:,:,0] = T[0]*R + T[3]*G + T[6]*B # X
    image[:,:,1] = T[1]*R + T[4]*G + T[7]*B # Y
    image[:,:,2] = T[2]*R + T[5]*G + T[8]*B # Z
    return image

def resize_image(im, size=(96,96), filter=Image.ANTIALIAS):
    '''Resize an image and return its array representation.

    Parameters
    ----------
    im : str, np.ndarray(uint8), or PIL.Image object
        The path to the image, an image array, or a loaded PIL.Image.
    size : tuple, (vdim, hdim)
        The desired output image size.

    Returns
    -------
    arr : uint8 np.array, (vdim, hdim, 3)
        The resized image array
    '''
    if isinstance(im, str):
        im = Image.open(im)
    elif isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    im.load()

    # flip to PIL.Image convention
    size = size[::-1]
    try:
        im = im._new(im.im.stretch(size, filter))
    except AttributeError:
        # PIL 4.0.0 The stretch function on the core image object has been removed.
        # This used to be for enlarging the image, but has been aliased to resize recently.
        im = im._new(im.im.resize(size, filter))
    im = np.asarray(im)
    return im

def _ff(Y):
    '''Obscure helper function
    '''
    fY = np.real(Y**(1./3.))
    idx = Y < 0.008856
    fY[idx] = Y[idx]*(7.787) + (4./29.)
    return fY

def _invgammacorrection(Rp):
    '''
    '''
    R = np.real(((Rp + 0.055)/1.055)**(2.4))
    idx = Rp < 0.04045
    R[idx] = Rp[idx] / 12.92
    return R

if __name__ == '__main__':
    #compute motion energy (ME) of a video. Functions borrowed from: https://github.com/gallantlab/pymoten
    #Here I reformulate how the ME of an array of video frames is computed. If the ME of a video frame array is
    # computed like their GitHub "synthetic video" example, the results will not match if the video is computed 
    # by being loaded as an .mp4. This is because they do not apply their specific rgb to luminance transforms
    # to "synthetic videos". Here, you can compute a videos ME from a video's array (e.g., npy file) or mp4 format.

    class arguments:
        def __init__(self) -> None:
            self.stim_root_mp4 = "/mnt/t/BOLDMoments/Nifti/stimuli"
            self.stim_root_npy = "/mnt/t/BOLDMoments/prepared_data/metadata/stimuli_npy"
            self.save_dir = "/mnt/t/BMDGeneration/analysis/metadata_features/motion_energy"
            self.compute_mp4 = True #whether to compute ME for mp4 video input. Just to make sure mp4 and npy inputs have some results
            self.compute_npy = False #whether to compute ME for npy video input. Just to make sure mp4 and npy inputs have some results
            self.duration = 3 #duration of each video in seconds. Needed to compute fps. Change to match your video
    args = arguments()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.compute_mp4:
        video_list_mp4 = glob.glob(os.path.join(args.stim_root_mp4, 'train', "*.mp4")) + glob.glob(os.path.join(args.stim_root_mp4, 'test', "*.mp4"))
        video_list_mp4.sort()

    if args.compute_npy:
        video_list_npy = glob.glob(os.path.join(args.stim_root_npy, "*.npy"))
        video_list_npy.sort()

    plot_dir = os.path.join(args.save_dir, "plots")
    if not os.path.exists( plot_dir):
        os.makedirs( plot_dir)

    for v in range(1000,1102):
        if args.compute_npy:
            #compute for npy
            vid_npy = np.load(video_list_npy[v]).astype(np.uint8) #shape (x, y, color, frames)
            vid_npy = np.moveaxis(vid_npy, 3, 0) #"project_stimulus" method expects frames first

            video_lum = imagearray2luminance(vid_npy, size=(96,96))
            numFrames, vdim, hdim = video_lum.shape
            fps = int(numFrames/args.duration)

            # Create a pyramid of spatio-temporal gabor filters
            pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=fps)
            features_npy = pyramid.project_stimulus(video_lum)

        if args.compute_mp4:
            #compute for mp4
            vid_mp4 = cv2.VideoCapture(video_list_mp4[v])
            assert(vid_mp4.isOpened())
            numFrames = int(vid_mp4.get(cv2.CAP_PROP_FRAME_COUNT))

            luminance_images = moten.io.video2luminance(video_list_mp4[v], size=(96,96), nimages=numFrames)
            nimages, vdim, hdim = luminance_images.shape
            assert(numFrames == nimages)
            fps = int(numFrames/args.duration)
            pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=fps)

            # Compute motion energy features
            features_mp4 = pyramid.project_stimulus(luminance_images)

        if args.compute_mp4 and args.compute_npy:
            assert((features_npy == features_mp4).all())
        elif args.compute_mp4 and not args.compute_npy:
            features = features_mp4
        elif args.compute_npy and not args.compute_mp4:
            features = features_npy

        videoID = f"vid_idx{v+1:04d}"

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.matshow(features, aspect='auto')
        plt.savefig(os.path.join(plot_dir, f"{videoID}_MEfeatures.png"))
        plt.clf()
        plt.close()

        np.save(os.path.join(args.save_dir, f"{videoID}_MotionEnergy.npy"), features)