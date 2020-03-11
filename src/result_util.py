import os
import numpy as np


def output_prediction_from_image(pred_scores, pred_boxes, img_w, img_h, class_name, output_dir, file_path):
    # define output file path
    base_name = file_path.split('/')[-1].replace('.npy', '.txt')
    output_file_path = os.path.join(output_dir, base_name)

    # output prediction
    with open(output_file_path, 'w') as fout:
        for score, box in zip(pred_scores, pred_boxes):
            left, top, right, bottom = int(box[1] * img_w), int(box[0] * img_h), int(box[3] * img_w), int(
                box[2] * img_h)
            fout.write('%s %f %d %d %d %d\n' % (class_name, float(score), left, top, right, bottom))


def output_prediction(scores, boxes, class_name, output_path):
    with open(output_path, "w") as fout:
        for b, s in zip(boxes, scores):
            ymin, xmin, ymax, xmax = b
            # output order: left top right bottom
            fout.write('{} {} {} {} {} {}\n'.format(
                class_name, s, int(xmin), int(ymin), int(xmax), int(ymax)
            ))


def output_feature_maps(feature_maps, filenames, output_dir):
    """
    filenames: shape [batch_size]
    feature_maps: numpy array, shape [batch_size, 32, 32, 128]
    output_dir: directory path to save .npy files
    """
    for feature_map, filename in zip(feature_maps, filenames):
        output_path = os.path.join(output_dir, str(filenames[0]).split('\'')[1].strip('.jpg'))
        np.save(output_path, feature_map)
