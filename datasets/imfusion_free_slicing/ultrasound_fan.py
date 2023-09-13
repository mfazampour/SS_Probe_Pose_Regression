import numpy as np

def create_ultrasound_mask(origin, opening_angle, short_radius, long_radius, img_shape):
    x, y = origin
    mask = np.zeros(img_shape)

    # Compute central point of the inner circle
    x = x - short_radius

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            dx, dy = i - x, j - y
            d = np.sqrt(dx*dx + dy*dy) - short_radius
            theta = np.arctan2(dy, dx) * (180 / np.pi)  # Convert to degrees

            if 0 <= d <= (long_radius - short_radius) and -opening_angle/2 <= theta <= opening_angle/2:
                mask[i, j] = 1

    # Calculate width and length
    width = 2 * long_radius * np.sin(np.radians(opening_angle / 2))
    length = long_radius - short_radius + origin[0]

    return mask, width, length


if __name__ == '__main__':
    origin = (106, 246)
    opening_angle = 70  # in degrees
    short_radius = 124  # in pixels
    long_radius = 512  # in pixels
    img_shape = (512, 512)
    mask, w, l = create_ultrasound_mask(origin, opening_angle, short_radius, long_radius, img_shape)

    print('Width: {} pixels'.format(w))
    print('Length: {} pixels'.format(l))

    # If you want to visualize the mask, you can use libraries like matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(mask, cmap='gray')
    plt.show()