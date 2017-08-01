import tensorflow as tf
class QueueReader:
    def __init__(self, filename_list, shared_name="queue", num_threads=16):
        self._filename_list = filename_list
        self._num_threads = num_threads

        # Filenames to name queue
        self._filename_queue = tf.train.string_input_producer(self._filename_list)
        self.coord = tf.train.Coordinator()
        self.shared_name = shared_name

    def batched(self, batch_size, min_after_dequeue):
        # csv file queue to one record 
        reader = tf.TextLineReader()
        key, value = reader.read(self._filename_queue)

        # record to batched
        capacity = min_after_dequeue + 4 * batch_size
        batched_record = tf.train.shuffle_batch([value],
                                     batch_size=batch_size,
                                     capacity=capacity,
                                     min_after_dequeue=min_after_dequeue,
                                     num_threads=self._num_threads,
                                     shared_name=self.shared_name)
        return batched_record 

    def start(self, session):
        threads = tf.train.start_queue_runners(sess=session, coord=self.coord)
        return threads, self.coord 
