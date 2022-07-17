# Big data pipeline for Deep learning - Tensorflow

## ETL: Extract, Transform, Load

- **Extraction** involves the process of extracting the data from multiple homogeneous or heterogeneous sources.

- **Transformation** refers to data cleansing and manipulation in order to convert them into a proper format.

- **Loading** is the injection of the transformed data into the memory of the processing units that will handle the training (whether this is CPUs, GPUs or even TPUs)

## Problems
- Data might not fit into memory.
- Data might not even fit into the local storage.
- Data might come from multiple sources.
- Utilize hardware as efficiently as possible both in terms of resources and idle time.
- Make processing fast so it can keep up with the accelerator’s speed.
- The result of the pipeline should be deterministic (or not).
- Being able to define our own specific transformations.
- Being able to visualize the process.

## Data Reading
- Loading from multiple sources: TFRecordDataset(S3), tfds,..
- Parallel data extraction: The **interleave()** function will load many data points concurrently and interleave the results so we don’t have to wait for each one of them to be loaded.

## Data Processing
- Get label
- Decode image
- Process path
- Augmentation for train dataset

## More explicit control of the training loop
- Iterators:  The big advantage of iterators is **lazy loading**. Instead of loading the entire dataset into memory, the iterator loads each data point only when it's needed.
- Batching: We apply all of our transformations on one batch at a time, avoiding to load all our data into memory at once.
- Prefetching: While the model is executing training step n, the input pipeline is reading the data for step n+1.
- Caching: Since each data point will be fed into the model more than once (one time for each epoch), why not store it into the memory? Save previous steps in the first epoch, from the second we just load from the cache.

## Data pipeline
- Read paths
- **SHUFFLE** with buffer size
- **MAP(augmentation)** with train dataset, **MAP(normal_process)** with validation and test dataset
- **Cache(Optional)**
- **Batching** with batch_size, repeat
- **Prefetch**
