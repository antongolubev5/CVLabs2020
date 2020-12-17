import cv2 as cv
import numpy as np
import pickle
import matplotlib.pyplot as plt


def show_image(img, title, gray):
    if gray:
        plt.imshow(img, 'gray')
    else:
        plt.imshow(img)

    plt.title(title)
    plt.show()


def prettify_depth_map(depth_map, kernel_blur, kernel_morpholohy):
    """
    https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    сглаживание, gauss_blur + морфологические преобразования
    erosion and dilation не очень
    """
    blur = cv.GaussianBlur(depth_map, kernel_blur, 0)
    # убираем артефакты
    close = cv.morphologyEx(blur, cv.MORPH_CLOSE, kernel_morpholohy)
    # заливка объекта
    final = cv.morphologyEx(close, cv.MORPH_OPEN, kernel_morpholohy)
    return final


def main():
    """
        ЗАДАНИЕ:
        Улучшение карты глубины - сгладить артефакты, убрать шум, улучшить качество

        изображение для того, чтобы понять что было и какие контуры потерялись
        морфология убрала некоторые линии которые должны быть
        в опенсв инструменты помогают детектить где должен быть прямоугольник

        Преобразование Хафа (кажется оно)
        наложить оригинал чтоб как то дорисовать примитивные линии фигуры (хз как)
        Дорисовать примитивные фигуры сходу (какие-то методы вроде есть , пункт выше наверное о том же)
        Морфология
    """

    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)

    original = data[0]
    depth_map = data[1]

    # бОльшие ядра для морфологии тк большие предметы на изображении
    prettified_depth_map = prettify_depth_map(depth_map, kernel_blur=(5, 5),
                                              kernel_morpholohy=np.ones((8, 8), np.uint8))

    show_image(original, 'original image', gray=True)
    show_image(depth_map, 'original depth map (original)', gray=False)
    show_image(depth_map, 'original depth map (gray)', gray=True)
    show_image(prettified_depth_map, 'prettified depth map', gray=True)


if __name__ == "__main__":
    main()
