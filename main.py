import tensorflow as tf
from configs.config import CFG
from utils.config import Config
from dataloader.dataloader import DataLoader

if __name__ == '__main__':
    config = Config.from_json(CFG)
    #data config
    train_path = config.data.train_path
    test_path = config.data.test_path
    class_names = config.data.class_names
    image_size = config.data.image_size
    #train config
    val_size = config.train.val_size
    batch_size = config.train.batch_size
    buffer_size = config.train.buffer_size

    rng = tf.random.Generator.from_seed(123, alg='philox')
    AUTOTUNE = tf.data.AUTOTUNE

    train_paths, val_paths, test_paths = DataLoader.file_paths(train_path, test_path, val_size)

    train_ds = DataLoader.configure_for_performance(train_paths, batch_size, buffer_size, class_names, image_size, rng, AUTOTUNE, training = True)
    val_ds = DataLoader.configure_for_performance(val_paths, batch_size, buffer_size, class_names, image_size, rng, AUTOTUNE, training = False)
    test_ds = DataLoader.configure_for_performance(test_paths, batch_size, buffer_size, class_names, image_size, rng, AUTOTUNE, training = False)
