import tensorflow as tf
import numpy as np

from src.constants import PARALLEL_ITERATIONS


def GraphExtractor(s_class_pred, s_boxes_pred, t_class_pred, t_boxes_pred,
                   iou_use=True, score_threshold=0.1, iou_threshold=0.6, max_boxes=20):
    """
    extract graph related consistency loss of mean teacher
    Argument:
        s_class_pred: class_predictions_with_background with shape (batch_size, num_anchors)
        s_boxes_pred: boxes with shape (batch_size, num_anchors, 4)
        t_class_pred: class_predictions_with_background with shape (batch_size, num_anchors)
        t_boxes_pred: boxes with shape (batch_size, num_anchors, 4)
        score_threshold: a float number.
        iou_threshold: a float number, threshold for IoU.
        max_boxes: an integer, maximum number of retained boxes.
    Returns:
        t_scores: a float tensor with shape [batch_size, num_boxes].
        t_boxes: a float tensor with shape [batch_size, num_boxes, 4].
        s_scores: a float tensor with shape [batch_size, num_boxes].
        s_boxes: a float tensor with shape [batch_size, num_boxes, 4].
    """
    
    def fn(x):
        s_score, s_box, t_score, t_box = x

        # low scoring boxes are removed
        ids = tf.where(tf.greater_equal(t_score, score_threshold))
        ids = tf.squeeze(ids, axis=1)
        t_box = tf.gather(t_box, ids)
        t_score = tf.gather(t_score, ids)
        s_box = tf.gather(s_box, ids)
        s_score = tf.gather(s_score, ids)

        if iou_use:
            selected_indices = tf.image.non_max_suppression(
                t_box, t_score, max_boxes, iou_threshold
            )
            t_box = tf.gather(t_box, selected_indices)
            t_score = tf.gather(t_score, selected_indices)
            s_box = tf.gather(s_box, selected_indices)
            s_score = tf.gather(s_score, selected_indices)
            num_boxes = tf.to_int32(tf.shape(t_box)[0])
        else:
            num_boxes = tf.to_int32(tf.shape(t_box)[0])

        zero_padding = max_boxes - num_boxes
        t_box = tf.pad(t_box, [[0, zero_padding], [0, 0]])
        t_score = tf.pad(t_score, [[0, zero_padding]])
        s_box = tf.pad(s_box, [[0, zero_padding], [0, 0]])
        s_score = tf.pad(s_score, [[0, zero_padding]])

        t_box.set_shape([max_boxes, 4])
        t_score.set_shape([max_boxes])
        s_box.set_shape([max_boxes, 4])
        s_score.set_shape([max_boxes])
        return s_score, s_box, t_score, t_box, num_boxes

    s_scores, s_boxes, t_scores, t_boxes, num_boxes = tf.map_fn(
        fn, [s_class_pred, s_boxes_pred, t_class_pred, t_boxes_pred],
        dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.int32),
        parallel_iterations=PARALLEL_ITERATIONS,
        back_prop=True, swap_memory=False, infer_shape=True
    )
    return s_scores, s_boxes, t_scores, t_boxes, num_boxes
