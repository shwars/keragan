import matplotlib.pyplot as plt
import numpy as np
from imutils import resize
import cv2

def show_images(images, cols = 1, titles = None):
    """
    Show a list of images using matplotlib
    :param images: list of images (or any sequence with len and indexing defined)
    :param cols: number of columns to use
    :param titles: list of titles to use or None
    """
    assert((titles is None)or (len(images) == len(titles)))
    if not isinstance(images,list):
        images = list(images)
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.xticks([]), plt.yticks([])
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.tight_layout()
    plt.show()

# code is adopted from https://medium.com/@Rakesh.thoppaen/image-preprocessing-without-changing-the-aspect-ratio-of-image-59c0afed70cb
def crop_resize(image,width,height,inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    dW = 0
    dH = 0
    # if the width is smaller than the height, then resize
    # along the width (i.e., the smaller dimension) and then
    # update the deltas to crop the height to the desired dimension
    if w < h:
        image = resize(image, width=width, inter=inter)
        dH = int((image.shape[0] - height) / 2.0)
    # otherwise, the height is smaller than the width so
    # resize along the height and then update the deltas
    # to crop along the width
    else:
        image = resize(image, height=height,inter=inter)
        dW = int((image.shape[1] - width) / 2.0)
    # now that our images have been resized, we need to
    # re-grab the width and height, followed by performing the crop
    (h, w) = image.shape[:2]
    image = image[dH:h - dH, dW:w - dW]
    # finally, resize the image to the provided spatial
    # dimensions to ensure our output image is always a fixed size
    return cv2.resize(image, (width, height),interpolation=inter)