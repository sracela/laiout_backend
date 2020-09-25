__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import numpy as np

import string
import random

class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        # from PIL import Image
        # image = Image.open(img_path)
        # new_image = image.resize((image_size, image_size))
        # im_arr = np.array(new_image)
        # img = im_arr.astype('float32')
        # img /= 255

        return img

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")
    
    @staticmethod
    def get_random_text(length_text=10, space_number=1, with_upper_case=True):
        results = []
        while len(results) < length_text:
            char = random.choice(string.ascii_letters[:26])
            results.append(char)
        if with_upper_case:
            results[0] = results[0].upper()

        current_spaces = []
        while len(current_spaces) < space_number:
            space_pos = random.randint(2, length_text - 3)
            if space_pos in current_spaces:
                break
            results[space_pos] = " "
            if with_upper_case:
                results[space_pos + 1] = results[space_pos - 1].upper()

            current_spaces.append(space_pos)

        return ''.join(results)

    @staticmethod
    def get_ios_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.digits + string.ascii_letters)
            results.append(char)

        results[3] = "-"
        results[6] = "-"

        return ''.join(results)

    @staticmethod
    def get_android_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.ascii_letters)
            results.append(char)

        return ''.join(results)
