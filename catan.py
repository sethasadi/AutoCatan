import cv2
import numpy as np
import copy
from tile import Tile


surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, nOctaves=4,
                                   nOctaveLayers=2, extended=True,
                                   upright=False)

resources = ['Lumber', 'Brick', 'Ore', 'Grain', 'Wool', 'Desert']

resource_images = []
resource_features = []
for resource in resources:
    resource_image = cv2.imread(f'{resource}.png')
    resource_images.append(resource_image)
    resource_features.append(surf.detectAndCompute(resource_image, None))

resource_dict = {}
for i, resource in enumerate(resources):
    resource_dict[i] = resource

frame_rate = 5

static_distributions = []

ip_stream_1 = '10.0.0.60'
ip_stream_2 = '10.0.0.93'

board_low = [80, 20, 100]
board_high = [220, 255, 255]
dir = 1

display_index = 0

gameboard = None

model_coords = []

ortho_width = 500
ortho_height = int(0.867 * ortho_width)
ortho_image_center = ortho_width // 2, ortho_height // 2
print(ortho_image_center)

game_model = np.float32([[5.75, 0], [16.25, 0], [21, 9.1], [16.25, 18.2],
                         [5.75, 18.2], [0, 9.1]])
game_model = np.multiply(game_model, ortho_width/float(21))


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


def calculate_static_distributions():
    global static_distributions
    static_distributions = []
    print('Generating static distributions')
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


def determine_resource(image, l2_thresh=0.5):
    dists = []
    for resource_feature in resource_features:
        good = find_good_matches(features_1=resource_feature, image_2=image)
        dists.append(len(good))
    l2_thresh = 0.5
    best_resource = None
    if sum(dists) != 0:
        dists = normalize(dists)
        for resource_name, dist in zip(resources, static_distributions):
            l2 = l2_norm(dists, dist)
            if l2 < l2_thresh:
                l2_thresh = l2
                best_resource = resource_name
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


def find_gameboard(stream):
    global gameboard
    confirmed = 0
    frame_number = 0
    approx = None
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
                    # cv2.drawContours(frame, [approx], -1, (10, 255, 0), 4)
                    confirmed += 1
                else:
                    # cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
                    pass
    print("Gameboard Identified")
    gameboard = approx


def get_orthophoto(frame):
    hex = np.zeros((6, 2), dtype=np.float32)
    for i, corner in enumerate(gameboard):
        hex[i] = corner[0]
    H, _ = cv2.findHomography(hex, game_model, cv2.RANSAC, 5.0)
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
       the second dimension is the corners, and the third dimension is x, y

       Order of the coords should be UL, UR, MR, LR, LL, ML'''
    tile_count = 19
    tile_coords = np.zeros((tile_count, 6, 2))

    center = (0, 0) + ortho_image_center

    tiles = {}
    tiles[center] = corners_from_center(center, tile_width)
    for first_surround in surrounding_tiles(center, tile_width):
        tiles[first_surround] = corners_from_center(first_surround, tile_width)
        for second_surround in surrounding_tiles(first_surround, tile_width):
            if second_surround in tiles.keys():
                continue
            tiles[second_surround] = corners_from_center(second_surround,
                                                         tile_width)

    assert len(tiles.keys()) == tile_count

    for i, tile in enumerate(tiles.keys()):
        for j, point in enumerate(tiles[tile]):
            for k, coord in enumerate(point):
                tile_coords[i, j, k] = coord


    return tile_coords


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

    return list(filter(lambda p: round(p + center), points))


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

    return list(filter(lambda p: round(p + center), points))


def get_search_area(tile):
    padding = 0
    start = tuple(tile[0] - padding)
    end = tuple(tile[3] + padding)
    return [start, end]


def check_for_buildings(ortho, corners):
    settlements = []
    cities = []
    for c in range(corners.shape[0]):
        corner = corners[c]
        padding = 3
        slice = ortho[corner[1] - padding:corner[1] + padding,
                      corner[0] - padding:corner[0] + padding]
        # check avg color of slice to determine if their is a settlement/city
        # settlement vs city will be difficult

    return settlements, cities


def determine_tile_number(tile):
    '''Find a circular contour and feature match for number
       TODO: use the thresh file to find how to thresh the circle out'''


if __name__ == "__main__":
    frame_number = 0

    stream = open_stream(ip_stream_1)
    calculate_static_distributions()
    if gameboard is None:
        find_gameboard(stream)

    while True:
        key = cv2.waitKey(33)
        handle_key(key, stream)
        frame = get_frame(stream, frame_number)
        frame_number += 1
        if frame is None:
            continue

        ortho = get_orthophoto(frame)
        cv2.imshow('Ortho', ortho)

        tiles = []

        # shape of corner coordinates should be (19, 6, 2)
        tile_coords = get_tile_coordinates()
        for t in range(tile_coords.shape[0]):
            # maybe make these points fit the ortho better with contours
            tile_corners = tile_coords[t]

            search = get_search_area(tile_corners)
            slice = ortho[search[0][1]:search[1][1], search[0][0]:search[1][0]]

            resource = determine_resource(slice, l2_thresh=5)

            # check all corners for settlements and cities
            settlements, cities = check_for_buildings(ortho)

            tile_number = determine_tile_number(slice)

            tile = Tile(tile_corners, slice, resource, settlements, cities,
                        tile_number)

            tiles.append(tile)

        # board = draw_gameboard(frame)
        # show_view(rgb=board)
        # resource = determine_resource(frame)
