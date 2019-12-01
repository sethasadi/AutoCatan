import cv2
import numpy as np
import copy
from tile import Tile
import pickle
import math


surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, nOctaves=4,
                                   nOctaveLayers=2, extended=True,
                                   upright=False)

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

# ip_stream_1 = '10.0.0.60'
ip_stream_1 = '10.0.0.60'
ip_stream_2 = '10.0.0.93'

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
game_model = np.multiply(game_model, ortho_width/float(21))

H = None

center_tile_corners = np.load('center_tile_corners.npy', allow_pickle=True)


# key codes:
# Shift: 0
# Tab: 9
# Shift-tab: 11
# Space: 32
# Enter: 13
# Delete: 8
# a-z: 97-122
# 0-9: 48-57


def find_good_matches(image_1=None, image_2=None, features_1=None,
                      features_2=None):
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


def live_update_resources(ortho):
    local_resource_count = {}
    for resource in resources:
        local_resource_count[resource] = 0

    tile_coords = get_tile_coordinates(1.8 * 500 / 21)

    for t in range(tile_coords.shape[0]):
        # maybe make these points fit the ortho better with contours

        tile_corners = tile_coords[t]

        search = get_search_area(tile_corners)
        slice = ortho[search[0][1]:search[1][1], search[0][0]:search[1][0]]

        zero_tile_corners = copy.copy(tile_corners)
        for c in zero_tile_corners:
            c[1] = c[1] - search[0][1]
            c[0] = c[0] - search[0][0]

        poly = np.zeros(slice.shape, dtype=np.uint8)
        poly = cv2.fillConvexPoly(poly, zero_tile_corners.astype('int32'), (255, 255, 255))
        slice = np.bitwise_and(poly, slice)

        cv2.imshow("Slice", slice)
        if cv2.waitKey(33) == 32:
            exit()

        resource_name = input('Enter resource name: ')
        cv2.imwrite(f'resources/{resource_name}_{local_resource_count[resource_name]}.png', slice)
        local_resource_count[resource_name] += 1


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
    weights = [2, 1, 1]
    return sum((w*(p-q))**2 for p, q, w in zip(list_1, list_2, weights)) ** .5


def determine_resource(image, l2_thresh=0.5):
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
    return best_resource
    # return resources[np.argmax(dists)]


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
    return frame


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


def determine_tile_number(tile):
    '''Find a circular contour and feature match for number
       TODO: use the thresh file to find how to thresh the circle out'''
    image_HSV = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    hue, saturation, value = list(cv2.split(image_HSV))

    _, hue_thresh_low = cv2.threshold(hue, 0, 255, cv2.THRESH_BINARY)
    _, hue_thresh_high = cv2.threshold(hue, 55, 255, cv2.THRESH_BINARY_INV)
    hue_thresh_combined = cv2.bitwise_and(hue_thresh_low, hue_thresh_high)
    cv2.imshow("hue", hue_thresh_combined)

    _, saturation_thresh_low = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY)
    _, saturation_thresh_high = cv2.threshold(saturation, 100, 255, cv2.THRESH_BINARY_INV)
    saturation_thresh_combined = cv2.bitwise_and(saturation_thresh_low, saturation_thresh_high)
    cv2.imshow("saturation", saturation_thresh_combined)

    _, value_thresh_low = cv2.threshold(value, 210, 255, cv2.THRESH_BINARY)
    _, value_thresh_high = cv2.threshold(value, 255, 255, cv2.THRESH_BINARY_INV)
    value_thresh_combined = cv2.bitwise_and(value_thresh_low, value_thresh_high)
    cv2.imshow("value", value_thresh_combined)

    img_result = cv2.bitwise_and(cv2.bitwise_and(hue_thresh_combined, saturation_thresh_combined), value_thresh_combined)

    # kernel = np.ones((7, 7), np.uint8)
    # img_result_close = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel)

    contours, hierachy = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        if circularity > 0.8:
            cv2.drawContours(slice, c, -1, (0, 140, 255), 2)
    cv2.imshow('Finding Tile', slice)
    while cv2.waitKey(33) != 32:
        pass


if __name__ == "__main__":
    frame_number = 0

    print("Generating Tile Coordinates")
    stream = open_stream(ip_stream_1)

    calculate_all_static_distributions()
    if gameboard is None:
        find_gameboard(stream)
    # calibrate_colors(stream, True)

    while True:
        key = cv2.waitKey(33)
        handle_key(key, stream)
        frame = get_frame(stream, frame_number)
        frame_number += 1
        if frame is None:
            continue

        ortho = get_orthophoto(frame)
        cv2.imshow('Ortho', ortho)

        # live_update_resources(ortho)

        tiles = []

        # shape of corner coordinates should be (19, 6, 2)
        tile_coords = get_tile_coordinates(1.8 * 500 / 21)

        for t in range(tile_coords.shape[0]):
            # maybe make these points fit the ortho better with contours

            tile_corners = tile_coords[t]

            search = get_search_area(tile_corners)
            slice = ortho[search[0][1]:search[1][1], search[0][0]:search[1][0]]

            zero_tile_corners = copy.copy(tile_corners)
            for c in zero_tile_corners:
                c[1] = c[1] - search[0][1]
                c[0] = c[0] - search[0][0]

            poly = np.zeros(slice.shape, dtype=np.uint8)
            poly = cv2.fillConvexPoly(poly, zero_tile_corners.astype('int32'), (255, 255, 255))
            slice = np.bitwise_and(poly, slice)
            cv2.imshow("Slice", slice)
            resource = determine_resource(slice, l2_thresh=.5)

            # check all corners for settlements and cities
            settlements, cities = check_for_buildings(ortho, tile_corners)
            print(f'Resource: {resource}')
            print(f'Settlements: {settlements}')
            print(f'Cities: {cities}')
            while cv2.waitKey(33) != 32:
                pass

            tile_number = determine_tile_number(slice)

            tile = Tile(tile_corners, slice, resource, settlements, cities,
                        tile_number)

            tiles.append(tile)
        ortho[ortho_image_center[1]][ortho_image_center[0]] = [0, 0, 255]
        cv2.imshow('Ortho', ortho)
        # while cv2.waitKey(33) != 32:
        #     pass

        # board = draw_gameboard(frame)
        # show_view(rgb=board)
        # resource = determine_resource(frame)
