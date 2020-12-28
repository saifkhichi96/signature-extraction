import os


def list_images(path, formats=['jpeg', 'jpg', 'png', 'tif', 'tiff']):
    """Lists all images in a directory (including sub-directories).

    Images in JPG, PNG and TIF format are listed.
    """
    images = []
    for f in os.listdir(path):
        fn = os.path.join(path, f)
        if os.path.isdir(fn):
            images += list_images(fn)
        else:
            ext = f.split('.')[-1]
            for format in formats:
                if ext.lower() == format.lower():
                    images.append(fn)
                    break

    return sorted(images)
