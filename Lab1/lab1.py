import numpy as np
import cv2 as cv


def rotate(image, angle):
    mat = np.zeros([2, 3])
    rad = np.deg2rad(angle)
    a = np.cos(rad)
    b = np.sin(rad)

    center_x = image.shape[1] / 2
    center_y = image.shape[0] / 2

    mat[0, 0] = a
    mat[0, 1] = b
    mat[1, 0] = -1 * b
    mat[1, 1] = a
    mat[0, 2] = (1 - a) * center_x - b * center_y
    mat[1, 2] = b * center_x + (1 - a) * center_y

    return cv.warpAffine(image, mat, image.shape[:2])


def flip(image, x, y):
    mat = np.zeros([2, 3])

    if x and y:
        return flip(flip(image, x=True, y=False), x=False, y=True)

    if x:
        mat[0, 0] = -1
        mat[1, 1] = 1
        mat[0, 2] = image.shape[1]

    if y:
        mat[0, 0] = 1
        mat[1, 1] = -1
        mat[1, 2] = image.shape[0]

    return cv.warpAffine(image, mat, image.shape[:2])


def translate(image, x, y):
    mat = np.zeros([2, 3])
    mat[0, 0] = 1
    mat[1, 1] = 1
    mat[0, 2] = x
    mat[1, 2] = y

    return cv.warpAffine(image, mat, image.shape[:2])


def scale(image, ax, ay):
    mat = np.zeros([2, 3])
    mat[0, 0] = ax
    mat[1, 1] = ay

    return cv.warpAffine(image, mat, image.shape[:2])


def main():
    """# C помощью афинных преобразований реализовать аугументацию изображения.
       # На вход принимается картинка, на выход несколько картинок:
       # - поворот исходного изображения
       # - симметричное отражение
       # - смещение исходного изображения
       # - сжатие и растяжение исходного изображения"""

    original = cv.imread('original.jpg')

    rotated = rotate(original, angle=45)
    cv.imwrite("rotated.jpg", rotated)

    flipped = flip(original, x=True, y=True)
    cv.imwrite("flipped.jpg", flipped)

    translated = translate(original, x=800, y=400)
    cv.imwrite("translated.jpg", translated)

    scaled = scale(original, ax=0.5, ay=0.8)
    cv.imwrite("scaled.jpg", scaled)

    print('Finished!')


if __name__ == "__main__":
    main()
