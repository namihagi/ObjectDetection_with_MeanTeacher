import os
import numpy as np


def output_prediction(pred_list, GT_boxes, GT_num_boxes, GT_filenames, result_dir, GT_dir):
    for pred, GT_box, GT_num_box, filename in zip(pred_list, GT_boxes, GT_num_boxes, GT_filenames):
        filename = filename.decode('utf-8')
        # prediction
        pred_path = os.path.join(result_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(pred_path, 'w') as fout:
            for idx, num_box in enumerate(pred['num_boxes']):
                for box_idx in range(num_box):
                    left, top, right, bottom = pred['boxes'][idx][box_idx][1], pred['boxes'][idx][box_idx][0], \
                                               pred['boxes'][idx][box_idx][3], pred['boxes'][idx][box_idx][2]
                    fout.write('%s %f %f %f %f %f\n' %
                               ('face', float(pred['scores'][idx][box_idx]),
                                float(left), float(top), float(right), float(bottom)))

        # groudtruth
        GT_path = os.path.join(GT_dir, str(filename).replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(GT_path, 'w') as fout:
            for box in GT_box:
                fout.write('%s %f %f %f %f\n' % ('face', float(box[1]), float(box[0]), float(box[3]), float(box[2])))


def output_prediction_from_image(pred_scores, pred_boxes, img_w, img_h, class_name, output_dir, file_path):
    # define output file path
    base_name = file_path.split('/')[-1].replace('.npy', '.txt')
    output_file_path = os.path.join(output_dir, base_name)

    # output prediction
    with open(output_file_path, 'w') as fout:
        for score, box in zip(pred_scores, pred_boxes):
            left, top, right, bottom = int(box[1]*img_w), int(box[0]*img_h), int(box[3]*img_w), int(box[2]*img_h)
            fout.write('%s %f %d %d %d %d\n' % (class_name, float(score), left, top, right, bottom))


def output_feature_maps(feature_maps, filenames, output_dir):
    """
    filenames: shape [batch_size]
    feature_maps: numpy array, shape [batch_size, 32, 32, 128]
    output_dir: directory path to save .npy files
    """
    for feature_map, filename in zip(feature_maps, filenames):
        output_path = os.path.join(output_dir, str(filenames[0]).split('\'')[1].strip('.jpg'))
        np.save(output_path, feature_map)
