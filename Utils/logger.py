#!/usr/bin/env python2
# -*- coding: utf-8 -*-"
# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc
from PIL import Image
from io import BytesIO, StringIO


# try:
#     from io import StringIO  # Python 2.7
# except ImportError:
#     from io import BytesIO         # Python 3.x


import tensorflow as tf

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        for i, img in enumerate(images):
            # Convert the image to an appropriate format
            img = (img * 255).astype(np.uint8)

            # Create a BytesIO buffer from the image
            output = BytesIO()
            Image.fromarray(img).save(output, format="PNG")
            image_string = output.getvalue()
            output.close()

            # Convert the image string to a tensor and add an extra dimension for the batch size
            image = tf.expand_dims(tf.image.decode_image(image_string), 0)

            # Log the image
            with self.writer.as_default():
                tf.summary.image(name=f"{tag}/{i}", data=image, step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()