import matplotlib.pyplot as plt
import numpy as np

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
