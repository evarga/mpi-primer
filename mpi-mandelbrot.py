"""
This script generates an image of the Mandelbrot set using the Message Passing Interface (MPI) for parallel computing.
The master process splits the work among several worker processes. Each worker process calculates a part of the image
and sends it back to the master process. The master process then assembles all the parts into the final image.

The script also includes functionality for overlaying images and saving them as an animated GIF. This animation is
created to show what parts of the final image were produced by each worker process. Looking at the distribution of work
across the processes can help identify potential performance bottlenecks and optimize the parallelization strategy.
For example, scheduling the work in a way that balances the load across processes can improve the overall performance.

Command line arguments can be used to specify the dimensions of the output image, the maximum number of iterations for
the Mandelbrot set computation, and the output filename.
"""
import argparse
from enum import IntEnum

import numpy as np
import imageio.v3 as iio
from mpi4py import MPI


class Color(IntEnum):
    BLACK = 0
    WHITE = 1


def in_mandelbrot_set(x, y, max_iteration):
    """Calculates whether the complex numbers x+iy are in the Mandelbrot set."""
    z = np.zeros(x.shape, dtype=np.complex64)
    iteration = np.zeros(x.shape, dtype=np.int32)
    for _ in range(max_iteration):
        # For inputs where most points lie outside the Mandelbrot set, the sequence
        # will diverge quickly for most points. As a result, the mask will select
        # fewer points with each iteration, and the function will perform fewer
        # computations for these points. On the other hand, for inputs where most
        # points lie inside or near the boundary of the Mandelbrot set, the sequence
        # may remain bounded for many iterations. In this case, the mask will continue
        # to select a large number of points with each iteration, and the function will
        # perform more computations for these points.
        #
        # This is why the processing time for each pixel can vary significantly, depending on
        # whether the corresponding point is inside or outside the Mandelbrot set. This issue
        # can be a challenge in parallel computing, as it can lead to load imbalance and
        # suboptimal performance. To address this issue, you can use load balancing techniques
        # such as dynamic scheduling, work stealing, or task migration. These techniques can
        # help distribute the work more evenly across the processes and improve the overall
        # performance of the parallel program. Nonetheless, due to involved complexity, it is
        # skipped in this educational unit.
        mask = z.real * z.real + z.imag * z.imag <= 4
        z[mask] = z[mask] * z[mask] + x[mask] + 1j * y[mask]
        iteration[mask] += 1
    return iteration == max_iteration


def index_to_coords(index, width):
    """Converts an index to the corresponding x, y coordinates."""
    return index % width, index // width


def create_image(start, end, width, height, max_iteration):
    """
    Creates an image of the Mandelbrot set for indices in the range of [start, end].

    Returns:
        numpy.ndarray: A packed bits array representing the portion of an image.
    """
    indices = np.arange(start, end)
    cols, rows = index_to_coords(indices, width)
    x = (cols / width - 0.5) * 2
    y = (rows / height - 0.5) * 2
    image = np.where(in_mandelbrot_set(x, y, max_iteration), Color.BLACK, Color.WHITE)
    # The next line is used to pack the bits of the binary image array into an 8-bit integer array.
    # This can be beneficial for reducing the amount of data that needs to be transferred between
    # processes in an MPI program, which can potentially improve performance.
    # However, the impact on performance will depend on several factors, including the size of the data,
    # the network bandwidth, and the time it takes to pack and unpack the bits.
    # If the time saved in data transfer is greater than the time spent on packing and unpacking,
    # then this operation can improve performance.
    #
    # It's also worth noting that packing the bits can reduce the memory footprint of the data,
    # which can be beneficial in scenarios where memory is a limiting factor.
    #
    # In conclusion, while packing can potentially improve the performance of an MPI program,
    # it's not guaranteed to do so in all cases. You should profile your specific application to
    # determine whether this operation is beneficial. It is done here for educational purposes.
    return np.packbits(image)


def parse_command_line_args():
    parser = argparse.ArgumentParser(
        description="Generates a Mandelbrot set image using the MPI framework.",
        epilog='MPI is based on the SPMD paradigm, which is a fundamental distributed computational pattern.')
    parser.add_argument(
        'width',
        metavar='X',
        type=int,
        help='The width of the output image in pixels.')
    parser.add_argument(
        'height',
        metavar='Y',
        type=int,
        help='The height of the output image in pixels.')
    parser.add_argument(
        '--output',
        metavar='filename',
        default='mandelbrot.gif',
        dest='output_filename',
        help='The name of the output image file saved as an animated GIF.')
    parser.add_argument(
        '--max_iterations',
        metavar='iterations',
        type=int,
        default=1000,
        dest='max_iterations',
        help='The max. number of iterations for deciding whether a pixel belongs to the Mandelbrot set or not.')
    return parser.parse_args()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_command_line_args()
    width = args.width
    height = args.height
    max_iteration = args.max_iterations

    total_pixels = width * height
    ranges = [(i * total_pixels // size, (i + 1) * total_pixels // size) for i in range(size)]
    range_parts = comm.scatter(ranges)

    image_part = create_image(*range_parts, width, height, max_iteration)
    gathered_data = comm.gather((range_parts, image_part))

    if rank == 0:
        # The master process is responsible for assembling the final image.
        base_image = np.empty((height, width, 3), dtype=np.uint8)
        overlay_image = np.empty((height, width, 3), dtype=np.uint8)
        overlay_image[:] = (0xF5, 0xFC, 0xFF)  # Light blue background.
        frames = [overlay_image.copy()]
        for i, (range_part, image_part) in enumerate(gathered_data):
            start, end = range_part
            indices = np.arange(start, end)
            cols, rows = index_to_coords(indices, width)

            # Convert the image part to black and white and add it to the base image.
            bw_image_part = np.unpackbits(image_part) * 255
            bw_image_part = np.stack([bw_image_part] * 3, axis=-1)
            base_image[rows, cols] = bw_image_part[indices - start]

            # Overlay the image part on the base image with a different shade of gray for each worker process.
            overlay_image_part = (np.unpackbits(image_part) + 1) * (i + 1) / size * 240
            overlay_image_part = np.stack([overlay_image_part] * 3, axis=-1)
            overlay_image[rows, cols] = overlay_image_part[indices - start]
            frames.append(overlay_image.copy())
        frames.append(base_image)
        iio.imwrite(args.output_filename, frames, duration=1700, loop=0)


if __name__ == "__main__":
    main()
