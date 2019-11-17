import os


def output_prediction(pred_list, GT_boxes, GT_num_boxes, GT_filenames, result_dir, GT_dir):
    for pred, GT_box, GT_num_box, filename in zip(pred_list, GT_boxes, GT_num_boxes, GT_filenames):

        # prediction
        pred_path = os.path.join(result_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(pred_path, 'w') as fout:
            for idx in range(pred['num_boxes']):
                left, top, right, bottom = pred[idx][2], pred[idx][1], pred[idx][4], pred[idx][3]
                fout.write('%s %f %f %f %f %f\n' % (filename, pred[idx]['scores'], left, top, right, bottom))

        # groudtruth
        GT_path = os.path.join(GT_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(GT_path, 'w') as fout:
            for idx in range(GT_num_box):
                fout.write('%s %f %f %f %f\n' % (filename, GT_box[2], GT_box[1], GT_box[4], GT_box[3]))
