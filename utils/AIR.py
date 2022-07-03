import os
import simplejson as json
import numpy as np


# Kinect v.2 camera coefficients
KINECT_CO = dict()
KINECT_CO['cfx'], KINECT_CO['cfy'] = 1144.361, 1147.337  # (mm)
KINECT_CO['cu0'], KINECT_CO['cv0'] = 966.359, 548.038
KINECT_CO['dfx'], KINECT_CO['dfy'] = 388.198, 389.033  # (mm)
KINECT_CO['du0'], KINECT_CO['dv0'] = 253.270, 213.934
KINECT_CO['k1'], KINECT_CO['k2'], KINECT_CO['k3'] = 0.108, -0.125, 0.062
KINECT_CO['p1'], KINECT_CO['p2'] = -0.001, -0.003
co = KINECT_CO


def read_joint(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fp:
            data = fp.read()
            body_info = json.loads(data)
    except:
        print('Error occurred: ' + path)

    return body_info


def vectorize(joint):
    return np.array([joint['x'], joint['y'], joint['z']]).astype('float32')


def move_camera_to_front(body_info, body_id):
    for f in range(len(body_info)):
        if body_info[f][body_id] is None:
            continue

        # joints of the trunk
        reference_body = body_info[f][body_id]["joints"]
        r_4_kinect = vectorize(reference_body[4])  # shoulderLeft
        r_8_kinect = vectorize(reference_body[8])  # shoulderRight
        r_20_kinect = (r_4_kinect + r_8_kinect) / 2  # spineShoulder
        r_0_kinect = vectorize(reference_body[0])  # torso
        dist_to_camera = np.linalg.norm(r_20_kinect)

        # find the front direction vector
        front_vector = np.cross(r_8_kinect - r_4_kinect, r_0_kinect - r_4_kinect)
        norm = np.linalg.norm(front_vector, axis=0, ord=2)
        norm = np.finfo(front_vector.dtype).eps if norm == 0 else norm
        normalized_front_vector = front_vector / norm * dist_to_camera
        cam_pos = r_20_kinect + normalized_front_vector  # to
        cam_dir = -normalized_front_vector
        if any(x != 0 for x in cam_dir):
            start_frame = f
            break

    # rotation factors
    eye = cam_pos
    at = cam_dir
    up = r_20_kinect - r_0_kinect

    norm_at = np.linalg.norm(at, axis=0, ord=2)
    norm_up = np.linalg.norm(up, axis=0, ord=2)
    norm_at = np.finfo(at.dtype).eps if norm_at == 0 else norm_at
    norm_up = np.finfo(up.dtype).eps if norm_up == 0 else norm_up
    z_c = at / norm_at
    y_c = up / norm_up
    x_c = np.cross(y_c, z_c)

    for f in range(start_frame):
        body = body_info[f][body_id]["joints"]
        for j in range(len(body)):
            body[j]['colorX'] = 0.
            body[j]['colorY'] = 0.

    for f in range(start_frame, len(body_info)):
        body = body_info[f][body_id]["joints"]
        for j in range(len(body)):
            joint = body[j]

            x = joint['x'] * x_c[0] + joint['y'] * x_c[1] + joint['z'] * x_c[2] - np.dot(eye, x_c)
            y = joint['x'] * y_c[0] + joint['y'] * y_c[1] + joint['z'] * y_c[2] - np.dot(eye, y_c)
            z = joint['x'] * z_c[0] + joint['y'] * z_c[1] + joint['z'] * z_c[2] - np.dot(eye, z_c)

            joint['x'] = x
            joint['y'] = y
            joint['z'] = z

            # project 3d point cloud to 2d depth map
            r2 = pow(x, 2) + pow(y, 2)
            x_corrected = x * (1 + co['k1'] * r2 + co['k2'] * pow(r2, 2) + co['k3'] * pow(r2, 3)) + \
                          2 * co['p1'] * x * y + co['p2'] * (r2 + 2 * pow(x, 2))
            y_corrected = y * (1 + co['k1'] * r2 + co['k2'] * pow(r2, 2) + co['k3'] * pow(r2, 3)) + \
                          co['p1'] * (r2 + 2 * pow(y, 2)) + 2 * co['p2'] * x * y

            joint['depthX'] = co['du0'] + co['dfx'] * x_corrected / z
            joint['depthY'] = co['dv0'] - co['dfy'] * y_corrected / z

            joint['colorX'] = co['cu0'] + co['cfx'] * x_corrected / z
            joint['colorY'] = co['cv0'] - co['cfy'] * y_corrected / z

    return body_info


def norm_features(body, method='vector'):
    if method != 'torso' and method != 'vector':
        raise Exception(f'Wrong normalization method: {method}')

    joints = get_upper_body_joints(body)
    if method == 'torso':
        return norm_to_torso(joints)
    if method == 'vector':
        return norm_to_vector(joints)


# move origin to torso and
# normalize to the distance between torso and spineShoulder
def norm_to_torso(joints):
    torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight = joints

    features = list()
    features.extend(norm_to_distance(torso, spineShoulder, spineShoulder))
    features.extend(norm_to_distance(torso, spineShoulder, head))
    features.extend(norm_to_distance(torso, spineShoulder, shoulderLeft))
    features.extend(norm_to_distance(torso, spineShoulder, elbowLeft))
    features.extend(norm_to_distance(torso, spineShoulder, wristLeft))
    features.extend(norm_to_distance(torso, spineShoulder, shoulderRight))
    features.extend(norm_to_distance(torso, spineShoulder, elbowRight))
    features.extend(norm_to_distance(torso, spineShoulder, wristRight))
    return features


def get_upper_body_joints(body):
    shoulderRight = vectorize(body[8])
    shoulderLeft = vectorize(body[4])
    elbowRight = vectorize(body[9])
    elbowLeft = vectorize(body[5])
    wristRight = vectorize(body[10])
    wristLeft = vectorize(body[6])

    torso = vectorize(body[0])
    spineShoulder = vectorize(body[20])
    head = vectorize(body[3])

    return torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight


def norm_to_distance(origin, basis, joint):
    norm = np.linalg.norm(basis - origin)
    norm = np.finfo(basis.dtype).eps if norm == 0 else norm
    result = (joint - origin) / norm if any(joint) else joint
    return result


def norm_to_vector(joints):
    torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight = joints

    features = list()
    features.extend(norm_to_distance(torso, spineShoulder, spineShoulder))
    features.extend(norm_to_distance(spineShoulder, head, head))
    features.extend(norm_to_distance(spineShoulder, shoulderLeft, shoulderLeft))
    features.extend(norm_to_distance(shoulderLeft, elbowLeft, elbowLeft))
    features.extend(norm_to_distance(elbowLeft, wristLeft, wristLeft))
    features.extend(norm_to_distance(spineShoulder, shoulderRight, shoulderRight))
    features.extend(norm_to_distance(shoulderRight, elbowRight, elbowRight))
    features.extend(norm_to_distance(elbowRight, wristRight, wristRight))
    return features


def denorm_features(features, method='vector'):
    if method != 'torso' and method != 'vector':
        raise Exception(f'Wrong normalization method: {method}')

    if method == 'torso':
        return denorm_from_torso(features)
    if method == 'vector':
        return denorm_from_vector(features)


def denorm_from_torso(features):
    # pelvis
    pelvis = np.array([0, 0, 0])

    # other joints (spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight)
    spine_len = 0.3
    features = np.array(features) * spine_len

    joints = np.vstack((pelvis, np.split(features, 8)))

    return joints


def denorm_from_vector(features):
    pelvis = np.array([0, 0, 0])
    v_spineShoulder, v_head, v_shoulderLeft, v_elbowLeft, v_wristLeft, v_shoulderRight, v_elbowRight, v_wristRight \
        = np.split(np.array(features), 8)

    spine_len = 0.3
    spineShoulder = v_spineShoulder * spine_len
    head = spineShoulder + v_head * (spine_len * 0.5)
    shoulderLeft = spineShoulder + v_shoulderLeft * (spine_len * 0.5)
    elbowLeft = shoulderLeft + v_elbowLeft * (spine_len * 0.6)
    wristLeft = elbowLeft + v_wristLeft * (spine_len * 0.5)
    shoulderRight = spineShoulder + v_shoulderRight * (spine_len * 0.5)
    elbowRight = shoulderRight + v_elbowRight * (spine_len * 0.6)
    wristRight = elbowRight + v_wristRight * (spine_len * 0.5)

    return np.vstack((pelvis, spineShoulder, head,
                      shoulderLeft, elbowLeft, wristLeft,
                      shoulderRight, elbowRight, wristRight))
