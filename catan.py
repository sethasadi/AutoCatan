import cv2
import numpy as np
import copy
from tile import Tile
from player import Player
import pickle
import math
import pyttsx3
from threading import Thread
import imutils
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import glob
from sklearn.model_selection import train_test_split
import time
import os

voice = pyttsx3.init()
speak_thread = None

surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, nOctaves=4,
                                   nOctaveLayers=2, extended=True,
                                   upright=False)
sift = cv2.xfeatures2d.SIFT_create()

resources = ['Lumber', 'Brick', 'Ore', 'Grain', 'Wool', 'Desert']
resources_all = []
resource_count = [4, 3, 3, 4, 4, 1]

file = open('color_dict_rgb.pkl', 'rb')
color_dict_rgb = pickle.load(file)
file.close()

resource_features = []


resource_dict = {}
for i, resource in enumerate(resources):
    resource_dict[i] = resource

frame_rate = 5

static_distributions = []
static_colors = []

# ip_stream_1 = '10.0.0.60'
ip_stream_1 = '10.0.0.60'
ip_stream_2 = '10.0.0.145'
ip_stream_3 = '10.0.0.93'

board_low = [80, 20, 100]
board_high = [150, 255, 255]
dir = 1

display_index = 0

gameboard = None

model_coords = []

ortho_width = 500
ortho_height = int(0.87 * ortho_width)
ortho_image_center = ortho_width // 2 + 12, ortho_height // 2
# ortho_image_center = 21 / 2, 18.2 / 2


game_model = np.float32([[5.75, 0], [16.25, 0], [21, 9.1], [16.25, 18.2],
                         [5.75, 18.2], [0, 9.1]])

game_model = np.float32([[5.75, 0], [0, 9.1], [5.75, 18.2], [16.25, 18.2],
                         [21, 9.1], [16.25, 0]])
game_model = np.multiply(game_model, ortho_width/float(21))

dice_model = np.float32([[0, 0], [50, 0], [50, 50], [0, 50]])

H = None

center_tile_corners = np.load('center_tile_corners.npy', allow_pickle=True)

tile_coords = np.load('hex_tile_coords.npy')

TILE_LOW = [5, 60, 210]
TILE_HIGH = [40, 120, 240]

tile_number_images = {}

tiles = []

try:
    dice_cnn_model = keras.models.load_model('dice_cnn_model.h5')
except IOError:
    dice_cnn_model = None

try:
    dice_x = np.load('dice_x.npy')
    dice_y = np.load('dice_y.npy')
    dice_x_train, dice_x_test, dice_y_train, dice_y_test = train_test_split(dice_x, dice_y, test_size=0.20, random_state=42)
except IOError:
    dice_x, dice_y, dice_x_train, dice_y_train, dice_x_test, dice_y_test = tuple([None]*6)

try:
    tile_num_cnn_model = keras.models.load_model('tile_num_cnn_model.h5')
except IOError:
    tile_num_cnn_model = None

try:
    tile_num_x = np.load('tile_num_x.npy')
    tile_num_y = np.load('tile_num_y.npy')
    tile_num_x_train, tile_num_x_test, tile_num_y_train, tile_num_y_test = train_test_split(tile_num_x, tile_num_y, test_size=0.20, random_state=42)
except IOError:
    tile_num_x, tile_num_y, tile_num_x_train, tile_num_y_train, tile_num_x_test, tile_num_y_test = tuple([None]*6)

tile_num_preds = []

tile_num_counts = [0]*13
correct_tile_num_counts = [0, 0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1]

players = []

os.system('clear')

# key codes:
# Shift: 0
# Tab: 9
# Shift-tab: 11
# Space: 32
# Enter: 13
# Delete: 8
# a-z: 97-122
# 0-9: 48-57


def white_balance(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mean_a = np.average(img_lab[:, :, 1])
    mean_b = np.average(img_lab[:, :, 2])
    img_lab[:, :, 1] = img_lab[:, :, 1] - ((mean_a - 128) * (img_lab[:, :, 0] / 255.0) * 1.1)
    img_lab[:, :, 2] = img_lab[:, :, 2] - ((mean_b - 128) * (img_lab[:, :, 0] / 255.0) * 1.1)
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    return img


def thread_say(message, vocalize=True):
    """Use pyttsx3 library to say a message, and also print to the
    console. If vocalize is false, just prints.
    """
    global speak_thread
    print(message)
    if vocalize:
        voice.say(message)
        if speak_thread is not None:
            speak_thread.join()
        speak_thread = Thread(target=voice.runAndWait, daemon=True)
        speak_thread.start()


def find_good_matches(image_1=None, image_2=None, features_1=None, features_2=None):

    if features_1 is None:
        features_1 = surf.detectAndCompute(image_1, None)
    if features_2 is None:
        features_2 = surf.detectAndCompute(image_2, None)
    if features_1[1] is None or features_2[1] is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(features_1[1], features_2[1], k=2)
    # go through and find the best matches
    good = []
    if matches is not None and len(matches[0]) == 2:
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
    return good


def update_resource_templates(stream):
    instructions = ('Press the following keys to save each specific resource '
                    + '(ESC to finish):')
    resource_strings = [resource for resource in map(
        lambda k: f'\n{k}: {resource_dict[k]}', resource_dict.keys())]
    instructions += ''.join(resource_strings)
    print(instructions)
    key = -1
    while key != 27:
        _, frame = stream.read()
        cv2.imshow('Resource Template Saving', frame)

        if key >= 49 and key <= 54:
            resource_index = key - 48
            resource = resource_dict[resource_index]
            print(f'Saving {resource}')
            cv2.imwrite(f'{resource}.png', frame)
        key = cv2.waitKey(33)


def get_masked_slice(ortho, tile_corners):
    search = get_search_area(tile_corners)
    slice = ortho[search[0][1]:search[1][1], search[0][0]:search[1][0]]

    poly = get_hex_mask(ortho, tile_corners)
    slice = np.bitwise_and(poly, slice)

    return slice


def get_hex_mask(ortho, tile_corners):
    search = get_search_area(tile_corners)
    slice = ortho[search[0][1]:search[1][1], search[0][0]:search[1][0]]

    zero_tile_corners = copy.copy(tile_corners)
    for c in zero_tile_corners:
        c[1] = c[1] - search[0][1]
        c[0] = c[0] - search[0][0]

    poly = np.zeros(slice.shape, dtype=np.uint8)
    poly = cv2.fillConvexPoly(poly, zero_tile_corners.astype('int32'), (255, 255, 255))

    return poly


def live_update_resources(ortho):
    global tile_coords

    local_resource_count = {}
    for resource in resources:
        local_resource_count[resource] = 0

    for t in range(tile_coords.shape[0]):
        tile_corners = tile_coords[t]

        slice = get_masked_slice(ortho, tile_corners)

        cv2.imshow("Slice", slice)
        if cv2.waitKey(33) == 32:
            exit()

        resource_name = input('Enter resource name: ')
        cv2.imwrite(f'resources/{resource_name}_{local_resource_count[resource_name]}.png', slice)
        local_resource_count[resource_name] += 1


def get_masked_tile_number(tile, visualize=False, get_mask=False):
    global TILE_LOW
    global TILE_HIGH

    image_HSV = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    hue, saturation, value = list(cv2.split(image_HSV))

    _, hue_thresh_low = cv2.threshold(hue, TILE_LOW[0], 255, cv2.THRESH_BINARY)
    _, hue_thresh_high = cv2.threshold(hue, TILE_HIGH[0], 255, cv2.THRESH_BINARY_INV)
    hue_thresh_combined = cv2.bitwise_and(hue_thresh_low, hue_thresh_high)

    _, saturation_thresh_low = cv2.threshold(saturation, TILE_LOW[1] , 255, cv2.THRESH_BINARY)
    _, saturation_thresh_high = cv2.threshold(saturation, TILE_HIGH[1], 255, cv2.THRESH_BINARY_INV)
    saturation_thresh_combined = cv2.bitwise_and(saturation_thresh_low, saturation_thresh_high)

    _, value_thresh_low = cv2.threshold(value, TILE_LOW[2], 255, cv2.THRESH_BINARY)
    _, value_thresh_high = cv2.threshold(value, TILE_HIGH[2], 255, cv2.THRESH_BINARY_INV)
    value_thresh_combined = cv2.bitwise_and(value_thresh_low, value_thresh_high)

    img_result = cv2.bitwise_and(cv2.bitwise_and(hue_thresh_combined, saturation_thresh_combined), value_thresh_combined)

    img_result_inv = cv2.bitwise_not(img_result)
    kernel = np.ones((3, 3), np.uint8)
    # img_result_inv = cv2.erode(img_result_inv, kernel)
    img_result_inv = cv2.morphologyEx(img_result_inv, cv2.MORPH_OPEN, kernel)

    if visualize:
        cv2.imshow("hue", hue_thresh_combined)
        cv2.imshow("saturation", saturation_thresh_combined)
        cv2.imshow("value", value_thresh_combined)
        cv2.imshow("result", img_result_inv)
        while cv2.waitKey(33) != 32:
            pass

    contours, hierachy = cv2.findContours(img_result_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    bounding_rect = None
    number_slice = None
    for c in contours:
        # cv2.drawContours(slice, c, -1, (255, 140, 0), 2)
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        if circularity > 0.5 and 200 < area < 600:
            bounding_rect = cv2.boundingRect(c)
            # cv2.drawContours(slice, c, -1, (0, 140, 255), 2)
    if bounding_rect is not None:
        x, y, w, h = bounding_rect
        if get_mask:
            entire_mask = np.full(tile.shape, 255, dtype=np.uint8)
            entire_mask = cv2.circle(entire_mask, (x + (w // 2), y + (h // 2)), w // 2, (0, 0, 0), -1)
            return entire_mask
        number_slice = tile[y:y+h, x:x+w]
        mask = np.zeros(number_slice.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (w // 2, h // 2), w // 2, (255, 255, 255), -1)

        number_slice = np.bitwise_and(number_slice, mask)
    elif get_mask:
        return np.full(tile.shape, 255, dtype=np.uint8)
    return number_slice


def live_update_tile_numbers(ortho, visualize=False):
    for t in range(tile_coords.shape[0]):
        tile_corners = tile_coords[t]
        tile = get_masked_slice(ortho, tile_corners)
        number_slice = get_masked_tile_number(tile, visualize)

        if number_slice is not None:
            cv2.imshow('Updating Tile Numbers', number_slice)
            if cv2.waitKey(33) == 32:
                exit()
            tile_name = int(input('Tile Number: '))
            cv2.imwrite(f'tile_numbers/{tile_name}/{datetime.now().time()}.png', number_slice)
        else:
            print('Skipping Desert')


def view_resource_templates():
    key = -1
    instructions = ('Press the following keys to view each specific resource '
                    + '(ESC to finish):')
    resource_strings = [resource for resource in map(
        lambda k: f'\n{k}: {resource_dict[k]}', resource_dict.keys())]
    instructions += ''.join(resource_strings)
    print(instructions)
    window = cv2.namedWindow("Resources")
    resource = None
    features = None
    features_showing = False
    while key != 27:
        if key >= 49 and key <= 54:
            resource_index = key - 48
            resource = cv2.imread(f'{resource_dict[resource_index]}.png')
            # cv2.imshow(window, resource)
        if key == 32:
            if features_showing:
                features_showing = False
            else:
                features_showing = True
                kp, des = surf.detectAndCompute(resource, None)
                features = cv2.drawKeypoints(resource, kp, None, (255, 0, 0),
                                             4)
        if features_showing and features is not None:
            cv2.imshow(window, features)
        elif resource is not None:
            cv2.imshow(window, resource)
        key = cv2.waitKey(33)


def calculate_all_static_distributions():
    print('Calculating all static distributions')
    global static_distributions
    global resource_features
    static_distributions = []
    for resource_name, count in zip(resources, resource_count):
        for i in range(count):
            resources_all.append(f'{resource_name}_{i}')

    for resource in resources_all:
        resource_image = cv2.imread(f'resources/{resource}.png')
        gray = cv2.cvtColor(resource_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(thresh, np.ones((12, 12), np.uint8))
        static_colors.append(cv2.mean(resource_image, mask))
        resource_features.append(surf.detectAndCompute(resource_image, None))

    for features_main in resource_features:
        dists = []
        for features_other in resource_features:
            good = find_good_matches(features_1=features_main,
                                     features_2=features_other)
            dists.append(len(good))
        dists = normalize(dists)
        static_distributions.append(dists)


def normalize(dists):
    dists_sum = sum(dists)
    for i in range(len(dists)):
        dists[i] = float(dists[i]) / float(dists_sum)
    assert round(sum(dists)) == 1
    return dists


def l2_norm(list_1, list_2):
    return sum((p-q)**2 for p, q in zip(list_1, list_2)) ** .5


def l2_norm_color(list_1, list_2):
    weights = [1, 1, 1]
    return sum((w*(p-q))**2 for p, q, w in zip(list_1, list_2, weights)) ** .5


def determine_resource(image, ortho, tile_corners, l2_thresh=0.5):
    dists = []
    for resource_feature in resource_features:
        good = find_good_matches(features_1=resource_feature, image_2=image)
        dists.append(len(good))
    l2_thresh = 0.5
    best_resource = None
    if sum(dists) != 0:
        dists = normalize(dists)
        # print(dists)
        for resource_name, dist in zip(resources_all, static_distributions):
            l2 = l2_norm(dists, dist)
            if l2 < l2_thresh:
                l2_thresh = l2
                best_resource = resource_name[:-2]

    hex_mask = get_hex_mask(ortho, tile_corners)
    hex_mask = cv2.erode(hex_mask, np.ones((12, 12), np.uint8))
    number_mask = get_masked_tile_number(image, get_mask=True)
    mask = np.bitwise_and(hex_mask, number_mask)
    # cv2.imshow('Mask', np.bitwise_and(mask, image))
    # while cv2.waitKey(33) != 32:
    #     pass
    avg_color = cv2.mean(image, cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    # print(l2_thresh, best_resource)
    l2_thresh = 500

    for resource_name, col in zip(resources_all, static_colors):
        l2 = l2_norm(avg_color, col)
        # print(f'{l2} from {resource_name}')
        if l2 < l2_thresh:
            l2_thresh = l2
            best_resource = resource_name[:-2]
    # print(l2_thresh, best_resource)
    return best_resource


def open_stream(ip):
    stream = cv2.VideoCapture()
    stream.open(f'http://{ip}/live')
    return stream


def approx_hexagon(corner_points):
    num_corners = len(corner_points)
    lengths = []
    for side in range(num_corners):
        x1, y1 = tuple(corner_points[side][0])
        x2, y2 = tuple(corner_points[(side + 1) % num_corners][0])
        lengths.append(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
    lengths = normalize(lengths)
    return l2_norm(lengths, [1/float(num_corners)]*num_corners)


def handle_key(key, stream):
    global dir
    global display_index
    global frame_rate
    if key == 27:
        exit(0)
    elif key == ord('q'):
        board_high[0] = min(board_high[0] + (5 * dir), 255)
    elif key == ord('w'):
        board_high[1] = min(board_high[1] + (5 * dir), 255)
    elif key == ord('e'):
        board_high[2] = min(board_high[2] + (5 * dir), 255)
    elif key == ord('a'):
        board_low[0] = max(board_low[0] - (5 * dir), 0)
    elif key == ord('s'):
        board_low[1] = max(board_low[1] - (5 * dir), 0)
    elif key == ord('d'):
        board_low[2] = max(board_low[2] - (5 * dir), 0)
    elif key == 0:
        dir *= -1
    elif key == 32:
        display_index = (display_index + 1) % 4
    elif key == 91:
        frame_rate = max(frame_rate - 1, 1)
    elif key == 93:
        frame_rate = frame_rate + 1
    elif key == ord('r'):
        find_gameboard(stream)


def show_view(rgb=None, hue=None, saturation=None, value=None):
    to_display = []
    if rgb is not None:
        to_display.append(rgb)
    if hue is not None:
        to_display.append(hue)
    if saturation is not None:
        to_display.append(saturation)
    if value is not None:
        to_display.append(value)
    cv2.imshow('Display', to_display[display_index % len(to_display)])


def find_gameboard(stream, visualize=False):
    print('Finding Gameboard')
    global gameboard
    confirmed = 0
    frame_number = 0
    approx = None
    good_approx = None
    while confirmed < 5:
        stream.grab()
        frame_number += 1
        if frame_number % frame_rate != 0:
            continue
        _, frame = stream.retrieve()
        image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = list(cv2.split(image_HSV))

        _, hue_thresh_low = cv2.threshold(hue, board_low[0], 255,
                                          cv2.THRESH_BINARY)
        _, hue_thresh_high = cv2.threshold(hue, board_high[0], 255,
                                           cv2.THRESH_BINARY_INV)
        hue_thresh_combined = cv2.bitwise_and(hue_thresh_low, hue_thresh_high)

        _, saturation_thresh_low = cv2.threshold(saturation, board_low[1], 255,
                                                 cv2.THRESH_BINARY)
        _, saturation_thresh_high = cv2.threshold(saturation, board_high[1],
                                                  255, cv2.THRESH_BINARY_INV)
        saturation_thresh_combined = cv2.bitwise_and(saturation_thresh_low,
                                                     saturation_thresh_high)

        _, value_thresh_low = cv2.threshold(value, board_low[2], 255,
                                            cv2.THRESH_BINARY)
        _, value_thresh_high = cv2.threshold(value, board_high[2], 255,
                                             cv2.THRESH_BINARY_INV)
        value_thresh_combined = cv2.bitwise_and(value_thresh_low,
                                                value_thresh_high)

        img_result = cv2.bitwise_and(hue_thresh_combined,
                                     saturation_thresh_combined)
        img_result = cv2.bitwise_and(img_result, value_thresh_combined)

        kernel = np.ones((3, 3), np.uint8)
        img_result = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel)
        img_result = cv2.morphologyEx(img_result, cv2.MORPH_OPEN, kernel)

        contours, hierachy = cv2.findContours(img_result, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_NONE)[-2:]
        if hierachy is None:
            continue
        for c, h in zip(contours, hierachy[0]):
            size = cv2.contourArea(c)
            if size > 40000 and h[3] == -1:
                epsilon = 0.03 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                corners = len(approx)
                hex_approx = approx_hexagon(approx)

                M_actual = cv2.moments(c)
                centroid_actual = [int(M_actual["m10"] / M_actual["m00"]),
                                   int(M_actual["m01"] / M_actual["m00"])]
                M_hex = cv2.moments(approx)
                centroid_hex = [int(M_hex["m10"] / M_hex["m00"]),
                                int(M_hex["m01"] / M_hex["m00"])]
                distance = ((centroid_actual[0] - centroid_hex[0]) ** 2 + (centroid_actual[1] - centroid_hex[1]) ** 2) ** 0.5
                if corners == 6 and hex_approx < 0.1 and distance < 5:
                    good_approx = approx
                    cv2.drawContours(frame, [approx], -1, (10, 255, 0), 4)
                    confirmed += 1
                else:
                    cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
                    pass
        if visualize:
            show_view(rgb=frame)
            if cv2.waitKey(33) == 27:
                break
    print("Gameboard Identified")
    gameboard = good_approx


def get_orthophoto(frame):
    global H
    hex = np.zeros((6, 2), dtype=np.float32)
    for i, corner in enumerate(gameboard):
        hex[i] = corner[0]
    H, _ = cv2.findHomography(hex, game_model)
    return cv2.warpPerspective(frame, H, (ortho_width, ortho_height))


def draw_gameboard(frame):
    return cv2.drawContours(copy.copy(frame), [gameboard], -1, (10, 255, 0), 4)


def get_frame(stream, frame_number):
    frame = None
    stream.grab()
    if frame_number % frame_rate == 0:
        _, frame = stream.retrieve()
    if frame is None:
        return None
    return white_balance(frame)


def get_tile_coordinates(tile_width):
    '''Return the coordinates of the corners of all of the tiles

       Should be of size n x 6 x 2 where the first dimension is the tiles,
       the second dimension is the corners, and the third dimension is x, y'''
    tile_count = 19
    tile_coords = np.zeros((tile_count, 6, 2))

    center = ortho_image_center
    i = 0
    tiles = {}
    tiles[center] = corners_from_center(center, tile_width)
    first_surrounds = surrounding_tiles(center, tile_width)
    for first_surround in first_surrounds:
        tiles[first_surround] = corners_from_center(first_surround, tile_width)
    for first_surround in first_surrounds:
        for second_surround in surrounding_tiles(first_surround, tile_width):
            already_present = False
            for prev in tiles.keys():
                if l2_norm(second_surround, prev) < tile_width/10:
                    already_present = True
                    break
            if already_present:
                continue
            tiles[second_surround] = corners_from_center(second_surround,
                                                         tile_width)

    assert len(tiles.keys()) == tile_count

    for i, tile in enumerate(tiles.keys()):
        for j, point in enumerate(tiles[tile]):
            tile_coords[i, j, :] = point

    return tile_coords.astype(np.uint16)


def corners_from_center(center, tile_width=1):
    """Returns the coordinates of all corners relative to a given center
    coordinate

    Order of the coords should be UL, U, UR, LR, L, LL
    """

    horizontal = ((3 ** 0.5) / 2.0) * tile_width
    vertical = tile_width
    half_vertical = tile_width / 2.0

    points = []
    points.append((-1 * horizontal, -1 * half_vertical))
    points.append((0.0, -1 * vertical))
    points.append((horizontal, -1 * half_vertical))
    points.append((horizontal, half_vertical))
    points.append((0.0, vertical))
    points.append((-1 * horizontal, half_vertical))

    offset_points = []
    for p in points:
        offset_points.append((p[0] + center[0], p[1] + center[1]))

    return offset_points


def surrounding_tiles(center, tile_width=1):
    """Gets coordinates of centers of all surrounding tiles given the center
    of a tile
    """

    horizontal = (3 ** 0.5) * tile_width
    half_horizontal = horizontal / 2.0
    vertical = 1.5 * tile_width

    points = []
    points.append((-1 * half_horizontal, -1 * vertical))
    points.append((half_horizontal, -1 * vertical))
    points.append((horizontal, 0.0))
    points.append((half_horizontal, vertical))
    points.append((-1 * half_horizontal, vertical))
    points.append((-1 * horizontal, 0.0))

    offset_points = []
    for p in points:
        offset_points.append((p[0] + center[0], p[1] + center[1]))

    return offset_points


def get_search_area(tile):
    padding = 10
    start = tuple(np.amin(tile, axis=0) - padding)
    end = tuple(np.amax(tile, axis=0) + padding)
    return [start, end]


def get_average_color(image, loc, radius, visualize=False):
    slice = image[loc[1] - radius:loc[1] + radius,
                  loc[0] - radius:loc[0] + radius]
    poly = np.zeros(slice.shape, dtype=np.uint8)
    poly = cv2.circle(poly, (radius, radius), radius, (255, 255, 255), -1)
    image_color = cv2.mean(slice, cv2.cvtColor(poly, cv2.COLOR_BGR2GRAY))
    max_i = max(image_color)
    image_color = [c for c in map(lambda i: i / max_i, image_color)][:3]
    if visualize:
        slice = np.bitwise_and(poly, slice)
        cv2.imshow('Average Color', slice)
        print(f'Average color is {image_color}')
        while cv2.waitKey(33) != 32:
            pass
    return image_color


def check_for_buildings(ortho, corners, visualize=False):
    settlements = []
    cities = []
    for corner in corners:
        padding = 3
        image_color = get_average_color(ortho, corner, padding, visualize)
        best_color = get_building_color(image_color)
        if visualize:
            print(best_color)
        if best_color is not None:
            settlements.append(best_color)
        # check avg color of slice to determine if their is a settlement/city
        # settlement vs city will be difficult
    return settlements, cities


def get_building_color(image_color):
    best_l2 = 1000
    best_color = None
    for color_name in color_dict_rgb.keys():
        l2 = l2_norm(image_color, color_dict_rgb[color_name])
        if l2 < best_l2:
            best_l2 = l2
            best_color = color_name
    return best_color


def calibrate_colors(stream, visualize=False):
    print('Calibrating colors')
    global color_dict_rgb
    frame_number = 0
    for color_name in color_dict_rgb.keys():
        print(f'Place six {color_name} buildings around the center tile and press SPACE')
        corners = []
        while cv2.waitKey(33) != 32:
            frame = get_frame(stream, frame_number)
            frame_number += 1
            if frame is None:
                continue
            ortho = get_orthophoto(frame)
            cv2.imshow('Color Calibration', ortho)
        for corner in center_tile_corners:
            corners.append(get_average_color(ortho, corner, 5, visualize))
        corners = np.array(corners)
        average = list(np.average(corners, axis=0))
        print('Average:', average)
        color_dict_rgb[color_name] = average
    f = open('color_dict_rgb.pkl', 'wb')
    pickle.dump(color_dict_rgb, f)
    f.close()


def load_tile_number_images():
    global tile_number_images

    tile_number_images_count = 0
    print('Loading tile number images')

    for num in range(13):
        images = glob.glob(f'tile_numbers/{num}/*.png')
        if len(images) != 0:
            tile_number_images[num] = []
        for filename in images:
            tile_number_images_count += 1
            image = cv2.imread(filename)
            image = cv2.resize(image, (25, 25))
            tile_number_images[num].append(image)
    print(f'Loaded {tile_number_images_count} tile number images')


def init_tile_num_training_data():
    global tile_num_x
    global tile_num_y
    global tile_num_x_train
    global tile_num_y_train
    global tile_num_x_test
    global tile_num_y_test

    step = 1
    tile_num_images = []
    tile_num_numbers = []
    tile_number_count = 0
    for num, image_list in zip(tile_number_images.keys(), tile_number_images.values()):
        for image in image_list:
            for angle in np.arange(0, 360, step):
                tile_number_count += 1
                tile_num_images.append(imutils.rotate(image, angle))
                tile_num_numbers.append(num - 2)

    # should be shape n x 50 x 50 x 3
    tile_num_x = np.zeros((tile_number_count, 25, 25, 3), dtype=np.uint8)
    # should be shape n x 1
    tile_num_y = np.zeros((tile_number_count,), dtype=np.uint8)

    assert len(tile_num_images) == len(tile_num_numbers) == tile_number_count != 0

    for i, image, number in zip(range(tile_number_count), tile_num_images, tile_num_numbers):
        tile_num_x[i, :, :, :] = image
        tile_num_y[i] = number
    assert tile_num_x.shape == (tile_number_count, 25, 25, 3)
    np.save('tile_num_x.npy', tile_num_x)
    np.save('tile_num_y.npy', tile_num_y)
    tile_num_x_train, tile_num_x_test, tile_num_y_train, tile_num_y_test = train_test_split(tile_num_x, tile_num_y, test_size=0.20, random_state=42)


def init_tile_num_cnn():
    global tile_num_cnn_model
    global tile_num_x_train
    global tile_num_y_train

    print('Initializing Tile Number CNN')

    tile_num_cnn_layers = [Conv2D(32, (3, 3), padding='same', input_shape=(25, 25, 3)), MaxPool2D((2, 2)), Conv2D(64, (3, 3), padding='same'), MaxPool2D((2, 2)), Conv2D(128, (3, 3), padding='same'), MaxPool2D((2, 2)), Conv2D(128, (3, 3), padding='same'), MaxPool2D((2, 2)), Flatten(), Dense(200, activation="relu"), Dense(200), Dense(11, activation="softmax")]
    tile_num_cnn_model = keras.Sequential(tile_num_cnn_layers)

    tile_num_cnn_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(tile_num_cnn_model.summary())

    tile_num_cnn_model.fit(tile_num_x_train, tile_num_y_train, epochs=3, verbose=2)

    tile_num_cnn_model.save('tile_num_cnn_model.h5')


def evaluate_tile_num_cnn():
    tile_num_cnn_scores = tile_num_cnn_model.evaluate(tile_num_x_test, tile_num_y_test)
    print(tile_num_cnn_scores)


def determine_tile_number(tile, visualize=False):
    global tile_num_preds
    global tile_num_counts

    tile_nums = []
    number_slice = get_masked_tile_number(tile, visualize)
    if number_slice is None:
        tile_num_preds.append(np.zeros((11)))
        return None
    if visualize:
        cv2.imshow('Tile', number_slice)
        while cv2.waitKey(33) != 32:
            pass
    number_slice = cv2.resize(number_slice, (25, 25))
    number_slice = number_slice.astype(np.float32)
    number_slice = number_slice.reshape(-1, 25, 25, 3)
    preds = None
    for i in range(5):
        preds = tile_num_cnn_model.predict(number_slice)
        tile_num_predict = preds.argmax() + 2

        # if tile_num_predict >= 11:
        #     print(f'11 score: {preds[0, 9]} and 12 score: {preds[0, 10]}')
        if visualize:
            print(f'Tile: {tile_num_predict}')
        tile_nums.append(tile_num_predict)
    tile_num_preds.append(preds[0])
    to_predict = max(set(tile_nums), key=tile_nums.count)
    # if to_predict == 11:
    #     to_predict = 12
    tile_num_counts[to_predict] += 1
    return to_predict

    step = 5
    number_slice = get_masked_tile_number(tile, visualize)
    if number_slice is None:
        return None
    number_slice = cv2.resize(number_slice, (25, 25))
    min_err = 10000000
    tile_num = -1
    for angle in np.arange(0, 360, step):
        rotated = imutils.rotate(number_slice, angle)
        for num, image_list in zip(tile_number_images.keys(), tile_number_images.values()):
            for image in image_list:
                err = np.sum((image - rotated) ** 2)
                if err < min_err:
                    min_err = err
                    tile_num = num
                if visualize:
                    cv2.imshow('Image 1', image)
                    cv2.imshow('Image 2', rotated)
                    print(num, err)
                    if cv2.waitKey(33) == 32:
                        exit()
    return tile_num


def generate_tiles(stream, visualize=False):
    global tiles
    global tile_num_counts

    tiles = []

    print('Generating Tiles')

    frame = None
    while frame is None:
        frame = get_frame(stream, 0)

    ortho = get_orthophoto(frame)

    if visualize:
        cv2.imshow('Tiles Orthophoto', ortho)
        while cv2.waitKey(33) != 32:
            pass

    for t in range(tile_coords.shape[0]):
        tile_corners = tile_coords[t]

        slice = get_masked_slice(ortho, tile_corners)
        resource = determine_resource(slice, ortho, tile_corners, l2_thresh=.5)

        # check all corners for settlements and cities
        settlements, cities = check_for_buildings(ortho, tile_corners)

        tile_number = determine_tile_number(slice, visualize)

        if tile_number is None:
            resource = 'Desert'

        if visualize:
            cv2.imshow("Slice", slice)
            thread_say(f'Resource: {resource}', vocalize=False)
            thread_say(f'Settlements: {settlements}', vocalize=False)
            thread_say(f'Cities: {cities}', vocalize=False)
            thread_say(f'Tile number: {tile_number}', vocalize=False)
            while cv2.waitKey(33) != 32:
                pass

        tile = Tile(tile_corners, slice, resource, settlements, cities,
                    tile_number)

        tiles.append(tile)

    # for tile in tiles:
    #     print(tile.tile_number)
    if sum(tile_num_counts) != 19:
        return
    while l2_norm(correct_tile_num_counts, tile_num_counts) != 0:
        # print(correct_tile_num_counts, tile_num_counts)
        for i, correct, actual in zip(range(13), correct_tile_num_counts, tile_num_counts):
            if actual < correct:
                # print(f'Not enough {i}\'s')
                vals = []
                for preds in tile_num_preds:
                    vals.append(copy.copy(preds[i-2]))
                # print(f'Length of vals: {len(vals)}')
                max_val = vals.index(max(vals))
                # print(f'Max val is {max_val}')
                while tiles[max_val].tile_number == i or tile_num_counts[tiles[max_val].tile_number] <= correct_tile_num_counts[tiles[max_val].tile_number]:
                    vals[max_val] = 0
                    max_val = vals.index(max(vals))
                    # print(f'Max val is {max_val}')
                # print(f'Changing tile {max_val}\'s')
                tile_num_counts[i] += 1
                tile_num_counts[tiles[max_val].tile_number] -= 1
                tiles[max_val].tile_number = i
            if actual != tile_num_counts[i]:
                break

    assert len(tiles) == 19
    print(f'{len(tiles)} Tiles Generated')


def get_dice(frame):
    global dice_model
    dice = np.zeros((2, 50, 50, 3), dtype=np.uint8)
    die_num = 0

    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = list(cv2.split(image_HSV))
    _, saturation_thresh = cv2.threshold(saturation, 100, 255, cv2.THRESH_BINARY_INV)
    _, value_thresh = cv2.threshold(value, 150, 255, cv2.THRESH_BINARY)

    result = np.bitwise_and(saturation_thresh, value_thresh)

    contours, hierachy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    if hierachy is None:
        return np.zeros((2, 50, 50, 3), dtype=np.uint8)
    for c, h in zip(contours, hierachy[0]):
        area = cv2.contourArea(c)
        if area > 1000 and area < 2000 and h[3] != -1:
            epsilon = 0.1 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if approx.shape[0] != 4:
                # print('Not a square, skipping...')
                continue
            H, _ = cv2.findHomography(approx, dice_model)
            die = cv2.warpPerspective(frame, H, (50, 50))
            if die_num > 1:
                return np.zeros((2, 50, 50, 3), dtype=np.uint8)
            dice[die_num, :, :, :] = die
            die_num += 1
    return dice


def add_role_images(stream):
    frame_number = 0
    number = int(input('Dice number to obtain: '))
    while True:
        frame = get_frame(stream_dice, frame_number)
        frame_number += 1
        if frame is None:
            continue
        cv2.imshow('Waiting for Dice', frame)
        key = cv2.waitKey(33)
        if key == 32:
            dice = get_dice(frame)
            for die in dice:
                if np.count_nonzero(die) == 0:
                    continue
                cv2.imshow('Die Image', die)
                if cv2.waitKey(33) == 27:
                    exit()
                cv2.imwrite(f'dice_rolls/{number}/{datetime.now().time()}.png', die)
                time.sleep(0.5)
                # roll = input('Dice Roll Value (n to not save, q to quit): ')
                # if roll == 'q':
                #     return
                # elif roll != 'n':
                #     cv2.imwrite(f'dice_rolls/{roll}/{datetime.now().time()}.png', die)
        elif key == 27:
            number = int(input('Dice number to obtain (0 to quit): '))
            if number == 0:
                return


def init_dice_training_data():
    global dice_x
    global dice_y
    global dice_x_train
    global dice_y_train
    global dice_x_test
    global dice_y_test
    print('Initialzing Training Data')

    image_count = 0
    dice_images = []
    dice_rolls = []
    for die_num in range(1, 7):
        for filename in glob.glob(f'dice_rolls/{die_num}/*.png'):
            image = cv2.imread(filename)
            for angle in range(0, 360, 90):
                image_count += 1
                dice_images.append(imutils.rotate(image, angle))
                dice_rolls.append(die_num - 1)
            image = np.flip(image)
            for angle in range(0, 360, 90):
                image_count += 1
                dice_images.append(imutils.rotate(image, angle))
                dice_rolls.append(die_num - 1)

    # should be shape n x 50 x 50 x 3
    dice_x = np.zeros((image_count, 50, 50, 3), dtype=np.uint8)
    # should be shape n x 1
    dice_y = np.zeros((image_count,), dtype=np.uint8)

    assert len(dice_images) == len(dice_rolls) == image_count != 0

    for i, image, number in zip(range(image_count), dice_images, dice_rolls):
        dice_x[i, :, :, :] = image
        dice_y[i] = number
    assert dice_x.shape == (image_count, 50, 50, 3)
    np.save('dice_x.npy', dice_x)
    np.save('dice_y.npy', dice_y)
    dice_x_train, dice_x_test, dice_y_train, dice_y_test = train_test_split(dice_x, dice_y, test_size=0.20, random_state=42)


def init_dice_cnn():
    global dice_cnn_model
    global dice_x_train
    global dice_y_train

    print('Initializing Dice CNN')

    dice_cnn_layers = [Conv2D(32, (3, 3), padding='same', input_shape=(50, 50, 3)), MaxPool2D((2, 2)), Conv2D(64, (3, 3), padding='same'), MaxPool2D((2, 2)), Conv2D(128, (3, 3), padding='same'), MaxPool2D((2, 2)), Conv2D(128, (3, 3), padding='same'), MaxPool2D((2, 2)), Flatten(), Dense(200, activation="relu"), Dense(200), Dense(6, activation="softmax")]
    dice_cnn_model = keras.Sequential(dice_cnn_layers)

    dice_cnn_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(dice_cnn_model.summary())

    dice_cnn_model.fit(dice_x_train, dice_y_train, epochs=30, verbose=2)

    dice_cnn_model.save('dice_cnn_model.h5')


def evaluate_dice_cnn():
    dice_cnn_scores = dice_cnn_model.evaluate(dice_x_test, dice_y_test)
    print(dice_cnn_scores)


def get_dice_roll(stream, visualize=False):
    # find dice
    dice_totals = []

    for i in range(5):
        frame = get_frame(stream, 0)

        dice = get_dice(frame)
        for die in dice:
            if np.count_nonzero(die) == 0:
                return None
        dice_cnn = dice.astype(np.float32)
        dice_cnn = dice_cnn.reshape(-1, 50, 50, 3)

        dice_total = 0
        dice_predict = dice_cnn_model.predict(dice_cnn)
        for probs in dice_predict:
            roll = probs.argmax() + 1
            if visualize:
                print(f'Die: {roll}')
            dice_total += roll
        if visualize:
            print(f'Total: {dice_total}')
        dice_totals.append(dice_total)
    return max(set(dice_totals), key=dice_totals.count)


def retrain_dice():
    init_dice_training_data()
    init_dice_cnn()
    evaluate_dice_cnn()


def retrain_tile_num():
    init_tile_num_training_data()
    init_tile_num_cnn()
    evaluate_tile_num_cnn()


if __name__ == "__main__":
    frame_number = 0
    thread_say("Welcome to Auto Catan! Initializing.", vocalize=False)

    two_phones = True
    if two_phones:
        stream = open_stream(ip_stream_3)
        stream_dice = open_stream(ip_stream_1)
    else:
        stream = open_stream(ip_stream_3)

    calculate_all_static_distributions()
    load_tile_number_images()
    if gameboard is None:
        find_gameboard(stream)

    # add_role_images(stream_dice)
    # retrain_dice()
    # live_update_tile_numbers(get_orthophoto(get_frame(stream, frame_number)))
    # load_tile_number_images()
    # retrain_tile_num()
    # live_update_resources(get_orthophoto(get_frame(stream, frame_number)))

    # calibrate_colors(stream)

    generate_tiles(stream)

    # for tile in tiles:
    #     print(f'Resource: {tile.resource}')
    #     print(f'Settlements: {tile.settlements}')
    #     print(f'Cities: {tile.cities}')
    #     print(f'Tile number: {tile.tile_number}')

    H_inv = np.linalg.inv(H)

    rolled = False
    height = 1

    while True:
        name = input('Input Player name (f to finish): ')
        if name == 'f':
            break
        color = input(f'Input {name}\'s color: ')
        players.append(Player(name, color))

    if two_phones:
        stream.release()
        stream_dice.release()
        stream = open_stream(ip_stream_3)
        stream_dice = open_stream(ip_stream_1)
    else:
        stream.release()
        stream = open_stream(ip_stream_3)

    print('\nBeginning Gameplay')
    none_count = 0
    while True:
        key = cv2.waitKey(33)
        handle_key(key, stream)
        if two_phones:
            frame = get_frame(stream, frame_number)
            dice_frame = get_frame(stream_dice, frame_number)
            frame_number += 1
            if dice_frame is None:
                continue
            roll = get_dice_roll(stream_dice)
            # ortho = get_orthophoto(frame)
            # cv2.imshow('Ortho', ortho)
            # if cv2.waitKey(33) == 27:
            #     exit()

        else:
            dice_frame = get_frame(stream, frame_number)
            frame_number += 1
            if dice_frame is None:
                continue
            roll = get_dice_roll(stream)

        if roll is not None and not rolled:
            print(f'\n{roll} Rolled')
            print('Disbursements:')
            dispersing = []
            for tile in tiles:
                dispersing = dispersing + tile.disperse(roll)
            for disperse in dispersing:
                for player in players:
                    if player.color == disperse[0]:
                        print(f'\t{player.name} gets {disperse[1]}')
            rolled = True
            none_count = 0
        elif roll is not None and rolled:
            none_count = 0
        elif roll is None and rolled:
            none_count += 1
            if none_count > 10:
                if two_phones:
                    generate_tiles(stream)
                rolled = False
