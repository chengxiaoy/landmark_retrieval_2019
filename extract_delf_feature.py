from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow_hub as hub
import os
from sklearn.externals import joblib

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.FATAL)

m = hub.Module('https://modeldepot.io/mikeshi/delf')

train_path = "test_data"


def get_image_list(train_path):
    training_names = os.listdir(train_path)

    image_paths = []
    for training_name in training_names:
        path = os.path.join(train_path, training_name)

        if os.path.isdir(path):
            image_paths.extend(get_image_list(path))
        else:
            if path.endswith(".jpg"):
                image_paths += [path]
    return image_paths


origin_image_paths = get_image_list(train_path)


def resize_image(origin_filename, filename, new_width=512, new_height=512):
    pil_image = Image.open(origin_filename)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert('RGB')
    pil_image_rgb.save(filename, format='JPEG', quality=90)


image_paths = []
for origin_image_path in origin_image_paths:
    steps = origin_image_path.split('/')
    image_paths.append(os.path.sep.join((train_path+"_256",steps[-1])))

for originfile, formatfile in zip(origin_image_paths, image_paths):
    resize_image(originfile, formatfile)


def image_input_fn():
    filename_queue = tf.train.string_input_producer(
        image_paths, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    return tf.image.convert_image_dtype(image_tf, tf.float32)


# The module operates on a single image at a time, so define a placeholder to
# feed an arbitrary image in.
image_placeholder = tf.placeholder(
    tf.float32, shape=(None, None, 3), name='input_image')

module_inputs = {
    'image': image_placeholder,
    'score_threshold': 100.0,
    'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
    'max_feature_num': 1000,
}

module_outputs = m(module_inputs, as_dict=True)

image_tf = image_input_fn()

with tf.train.MonitoredSession() as sess:
    results_dict = {}  # Stores the locations and their descriptors for each image
    for image_path in image_paths:
        image = sess.run(image_tf)
        print('Extracting locations and descriptors from %s' % image_path)
        results_dict[image_path] = sess.run(
            [module_outputs['locations'], module_outputs['descriptors']],
            feed_dict={image_placeholder: image})

descriptors_map = {}
locations_map = {}

for image_path in image_paths:
    locations_1, descriptors_1 = results_dict[image_path]
    descriptors_map[image_path] = descriptors_1
    locations_map[image_path] = locations_1

# 0 代表不压缩
joblib.dump((descriptors_map, image_paths, locations_map), "delf.pkl", compress=0)
