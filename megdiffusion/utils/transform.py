def linear_scale(image):
    """Uint8 [0, 255] -> Float [-1, 1]"""
    return (image / 255 * 2) - 1
 
def linear_scale_rev(image):
    """Float [-1, 1] -> Uint8 [0, 255]"""
    return ((image + 1) / 2 * 255).astype("uint8")