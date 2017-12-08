#conding = utf-8

import numpy as np
import cv2
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import utils

def get_bbox(cont_img, rgb_image=None):
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = None
    if rgb_image is not None:
        rgb_contour = rgb_image.copy()
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes, rgb_contour


def find_roi_bbox(self, rgb_image):
    # hsv -> 3 channel
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red) #lower20===>0,upper200==>255

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    bounding_boxes, rgb_contour = self.get_bbox(image_open, rgb_image=rgb_image)
    return bounding_boxes, rgb_contour, image_open

def find_roi_bbox_tumor_gt_mask( mask_image):
    mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    bounding_boxes, _ = get_bbox(np.array(mask))
    return bounding_boxes

def read_wsi_tumor(wsi_path, mask_path):
    try:
        wsi_image = OpenSlide(wsi_path)

        level_used = 8

        rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                   wsi_image.level_dimensions[level_used]))
#        resize_factor = float(1.0 / pow(2,5))
#        print('resize_factor: %f' % resize_factor) 
        
        tumor_gt_mask = cv2.imread(mask_path)
        tumor_gt_mask = np.array(tumor_gt_mask)
    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None, None

    return wsi_image, rgb_image,  tumor_gt_mask, level_used

def extract_positive_patches_from_tumor_region(wsi_image, tumor_gt_mask, level_used,
                                               bounding_boxes, patch_save_dir, patch_prefix,
                                               patch_index):
    """

        Extract positive patches targeting annotated tumor region

        Save extracted patches to desk as .png image files

        :param wsi_image:
        :param tumor_gt_mask:
        :param level_used:
        :param bounding_boxes: list of bounding boxes corresponds to tumor regions
        :param patch_save_dir: directory to save patches into
        :param patch_prefix: prefix for patch name
        :param patch_index:
        :return:
    """
    mag_factor = pow(2, level_used)
    tumor_gt_mask = cv2.cvtColor(tumor_gt_mask, cv2.COLOR_BGR2GRAY)
    print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))
    print('The mag_factor us:%d',mag_factor)
    for bounding_box in bounding_boxes:
        b_x_start = int(bounding_box[0])
        b_y_start = int(bounding_box[1])
        b_x_end = int(bounding_box[0]) + int(bounding_box[2])
        b_y_end = int(bounding_box[1]) + int(bounding_box[3])
#        X = np.random.random_integers(b_x_start, high=b_x_end, size=500 )
#        Y = np.random.random_integers(b_y_start, high=b_y_end, size=int((b_y_end-b_y_start)//2+1 ))
        col_cords = np.arange(b_x_start, b_x_end)
        row_cords = np.arange(b_y_start, b_y_end)
#        for x, y in zip(X, Y):
#            if int(tumor_gt_mask[y, x]) != 0:
        for x in col_cords:
            for y in row_cords:
                if int(tumor_gt_mask[y, x]) != 0:
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    patch = wsi_image.read_region((x_large, y_large), 0, (utils.PATCH_SIZE, utils.PATCH_SIZE))
                    patch.save(patch_save_dir + patch_prefix + str(patch_index), '.PNG')
                    patch_index += 1
                    patch.close()

    return patch_index

wsi_image, rgb_image,  tumor_gt_mask, level_used = read_wsi_tumor(utils.WSI_PATCH, utils.TUMOR_PATCH)

#tumor_gt_mask =  cv2.imread(TUMOR_PATCH)

bounding_boxes = find_roi_bbox_tumor_gt_mask(np.array(tumor_gt_mask))

patch_index = extract_positive_patches_from_tumor_region(wsi_image, np.array(tumor_gt_mask),
                                                                         level_used, bounding_boxes,
                                                                         utils.patch_save_dir, utils.patch_prefix,
                                                                         utils.patch_index)
wsi_image.close()
