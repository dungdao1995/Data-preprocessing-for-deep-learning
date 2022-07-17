import tensorflow as tf
import pathlib
import os

class DataLoader:

    @staticmethod
    def file_paths(train_path, test_path, val_size):
        """Loads dataset from path"""
        #train config
        train_root = train_path
        train_root = pathlib.Path(train_root)
        list_train = tf.data.Dataset.list_files(str(train_root/'*'))
        #train, validation
        image_count = len(list_train)
        val_size = int(image_count * val_size)
        train_paths = list_train.skip(val_size)
        val_paths = list_train.take(val_size)

        #test config
        test_root = test_path
        test_root = pathlib.Path(test_root)
        test_paths = tf.data.Dataset.list_files(str(test_root/'*'))

        return train_paths, val_paths, test_paths

    @staticmethod
    def _get_label(file_path, class_names):
        # Convert the path to a list of path components
        file = tf.strings.split(file_path, os.sep)[1]
        label = tf.strings.split(file,'.')[0]
        one_hot = label == class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    @staticmethod
    def _decode_img(image,image_size):
        image = tf.io.decode_jpeg(image, channels = 3)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
        return image

    @staticmethod
    def _process_path(file_path,class_names,image_size):
        label = DataLoader._get_label(file_path, class_names)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = DataLoader._decode_img(img, image_size)
        return img, label

    @staticmethod
    def _augment(file_path, class_names, image_size, seed):
        image, label = DataLoader._process_path(file_path,class_names,image_size)
        image = tf.image.resize_with_crop_or_pad(image, image_size + 6, image_size + 6)
        # Random crop back to the original size.
        image = tf.image.stateless_random_crop(
            image, size=[image_size, image_size, 3], seed=seed)
        # Random brightness.
        image = tf.image.stateless_random_brightness(
            image, max_delta=0.5, seed=seed)
        image = tf.clip_by_value(image, 0, 1)
        return image, label

    # Create a wrapper function for updating seeds.
    @staticmethod
    def _f(file_path, class_names, image_size, rng):
        #generate new seed
        seed = rng.make_seeds(2)[0]
        image, label = DataLoader._augment(file_path, class_names, image_size, seed)
        return image, label

    @staticmethod
    def configure_for_performance(ds, batch_size, buffer_size, class_names, image_size, rng, AUTOTUNE, training = False):
        if training:
            #Shuffle
            ds = ds.shuffle(buffer_size)
            # Augmentation for training dataset
            ds = ds.map(lambda ds: DataLoader._f(ds,class_names, image_size, rng),
                        num_parallel_calls=AUTOTUNE)
            # Batch all datasets.
            ds = ds.batch(batch_size).repeat()
        else:
            ds = ds.map(lambda ds: DataLoader._process_path(ds,class_names,image_size),
                        num_parallel_calls=AUTOTUNE)
            # Batch all datasets.
            ds = ds.batch(batch_size).repeat()

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)


