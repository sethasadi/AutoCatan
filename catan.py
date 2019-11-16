import cv2


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

frame_rate = 10

static_distributions = []

# key codes:
# Shift: 0
# Tab: 9
# Shift-tab: 11
# Space: 32
# Enter: 13
# Delete: 8
# a-z: 97-122
# 0-9: 48-57


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
                features = cv2.drawKeypoints(resource, kp, None, (255, 0, 0), 4)
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
            good = find_good_matches(features_1=features_main, features_2=features_other)
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


if __name__ == "__main__":
    ip_stream_1 = '10.0.0.60'
    ip_stream_2 = '10.0.0.93'

    stream_1 = cv2.VideoCapture()
    stream_2 = cv2.VideoCapture()

    stream_1.open(f'http://{ip_stream_1}/live')
    stream_2.open(f'http://{ip_stream_2}/live')
    frame_number = 0

    calculate_static_distributions()

    while cv2.waitKey(33) != 27:
        _, frame = stream_1.read()
        frame_number += 1
        if frame_number % frame_rate != 0:
            continue
        cv2.imshow('Working', frame)
        dists = []
        for resource_feature in resource_features:
            good = find_good_matches(features_1=resource_feature, image_2=frame)
            dists.append(len(good))
        max_l2 = 0.5
        best_resource = None
        if sum(dists) != 0:
            dists = normalize(dists)
            for resource_name, dist in zip(resources, static_distributions):
                l2 = l2_norm(dists, dist)
                if l2 < max_l2:
                    max_l2 = l2
                    best_resource = resource_name
        print(best_resource)
