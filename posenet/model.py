import tensorflow as tf
import os
import posenet.converter.config

MODEL_DIR = './_models'
DEBUG_OUTPUT = False


def model_id_to_ord(model_id):
    if 0 <= model_id < 4:
        return model_id  # id is already ordinal
    elif model_id == 50:
        return 0
    elif model_id == 75:
        return 1
    elif model_id == 100:
        return 2
    else:  # 101
        return 3


def load_config(model_ord):
    converter_cfg = posenet.converter.config.load_config()
    checkpoints = converter_cfg['checkpoints']
    output_stride = converter_cfg['outputStride']
    checkpoint_name = checkpoints[model_ord]

    model_cfg = {
        'output_stride': output_stride,
        'checkpoint_name': checkpoint_name,
    }
    return model_cfg


def load_model(model_id, sess, model_dir=MODEL_DIR):
    model_ord = model_id_to_ord(model_id)
    model_cfg = load_config(model_ord)
    model_path = os.path.join(model_dir, 'model-%s' % model_cfg['checkpoint_name'])
    if not os.path.exists(model_path):
        print('Cannot find model file %s, converting from tfjs...' % model_path)
        from posenet.converter.tfjs2python import convert
        convert(model_ord, model_dir, check=False)
        assert os.path.exists(model_path)

    sess.graph.as_default()
    tf.compat.v1.saved_model.loader.load(sess, ["serve"], model_path)

    # if DEBUG_OUTPUT:
    #     graph_nodes = [n for n in graph_def.node]
    #     names = []
    #     for t in graph_nodes:
    #         names.append(t.name)
    #         print('Loaded graph node:', t.name)

    offsets = sess.graph.get_tensor_by_name('offset_2:0')
    displacement_fwd = sess.graph.get_tensor_by_name('displacement_fwd_2:0')
    displacement_bwd = sess.graph.get_tensor_by_name('displacement_bwd_2:0')
    heatmaps = sess.graph.get_tensor_by_name('heatmap:0')

    return model_cfg, [heatmaps, offsets, displacement_fwd, displacement_bwd]
