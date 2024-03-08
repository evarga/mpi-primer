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
from PIL import Image
import imageio.v3 as iio
from mpi4py import MPI


class Color(IntEnum):
    BLACK = 0
    WHITE = 1


def in_mandelbrot_set(x, y):
    """Calculates whether the complex numbers x+iy are in the Mandelbrot set."""
    global max_iteration

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
        # such as dynamic scheduling, work stealing, task migration, or overlapping computation 
        # with communication. These techniques can help distribute the work more evenly across 
        # the processes and improve the overall performance of the parallel program.
        mask = z.real * z.real + z.imag * z.imag <= 4
        z[mask] = z[mask] * z[mask] + x[mask] + 1j * y[mask]
        iteration[mask] += 1
    return iteration == max_iteration


def index_to_coords(index):
    """Converts an index to the corresponding x, y coordinates."""
    global width

    return index % width, index // width


def create_image(start, end):
    """
    Creates an image of the Mandelbrot set for indices in the range of [start, end].

    Returns:
        numpy.ndarray: A packed bits array representing the portion of an image.
    """
    global width, height

    indices = np.arange(start, end)
    cols, rows = index_to_coords(indices)
    x = (cols / width - 0.5) * 2
    y = (rows / height - 0.5) * 2
    image = np.where(in_mandelbrot_set(x, y), Color.BLACK, Color.WHITE)

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
    parser.add_argument(
        '--schedule',
        metavar='schedule',
        default='static',
        choices=['static', 'dynamic'],
        dest='schedule',
        help='The scheduling policy.')
    return parser.parse_args()


def assemble_image(gathered_data):
    global args, width, height

    base_image = np.empty((height, width, 3), dtype=np.uint8)
    overlay_image = np.empty((height, width, 3), dtype=np.uint8)
    for chunk, image_part, i in gathered_data:
        start, end = chunk
        indices = np.arange(start, end)
        cols, rows = index_to_coords(indices)

        # Convert the image part to black and white and add it to the base image.
        bw_image_part = np.unpackbits(image_part) * 255
        bw_image_part = np.stack([bw_image_part] * 3, axis=-1)
        base_image[rows, cols] = bw_image_part[indices - start]

        # Overlay the image part on the base image with a different shade of gray for each worker process.
        overlay_image_part = np.ones(end - start) * (i + 1) / size * 240
        overlay_image_part = np.stack([overlay_image_part] * 3, axis=-1)
        overlay_image[rows, cols] = overlay_image_part[indices - start]

    # Convert numpy arrays to PIL images and blend them with an alpha value.
    base_image = Image.fromarray(base_image.astype(np.uint8))
    overlay_image = Image.fromarray(overlay_image.astype(np.uint8))
    blended_image = Image.blend(overlay_image, base_image, alpha=0.3)

    frames = [base_image, blended_image]
    iio.imwrite(args.output_filename, frames, duration=3500)


def static():
    global comm, rank, size, total_pixels

    ranges = [(i * total_pixels // size, (i + 1) * total_pixels // size) for i in range(size)]    
    local_range = comm.scatter(ranges, root=0)

    image_part = create_image(*local_range)

    gathered_data = comm.gather((local_range, image_part, rank), root=0)

    if rank == 0:
        assemble_image(gathered_data)



def dynamic():
    global comm, rank, size, total_pixels

    chunk_size = total_pixels // (size * 10)  # The chunk size is set to 10% of the total number of pixels.
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, total_pixels)) for i in range(total_pixels // chunk_size + 1)]
    chunk_indices = list(range(len(chunks)))
    workers_size = 0

    if rank == 0:
        # The master process distributes the first chunk to each worker process.
        for i in range(1, size):
            if chunk_indices:
                workers_size += 1
                chunk_index = chunk_indices.pop()
                comm.send(chunks[chunk_index], dest=i)

        # The master process receives the results from the worker processes and distributes the remaining chunks.
        # Furthermore, the master is also playing the role of a worker while waiting results to arrive.
        gathered_data = []
        while chunk_indices or workers_size > 0:
            if chunk_indices and not comm.Iprobe(source=MPI.ANY_SOURCE):
                chunk_index = chunk_indices.pop()
                chunk = chunks[chunk_index]
                image_part = create_image(*chunk)
                gathered_data.append((chunk, image_part, 0))

            if comm.Iprobe(source=MPI.ANY_SOURCE):
                status = MPI.Status()
                chunk, image_part = comm.recv(source=MPI.ANY_SOURCE, status=status)
                gathered_data.append((chunk, image_part, status.Get_source()))

                if chunk_indices:
                    chunk_index = chunk_indices.pop()
                    comm.send(chunks[chunk_index], dest=status.Get_source())
                else:
                    # Each time we receive something from a worker and cannot give it a new task, then
                    # we must decrement the number of active workers.
                    workers_size -= 1

        # Send a None value to each worker process to signal that all tasks are completed.
        for i in range(1, size):
            comm.send(None, dest=i)

        assemble_image(gathered_data)
    else:
        while True:
            # The worker processes receive a chunk, process it, and send the result back to the master process.
            chunk = comm.recv(source=0)
            # A None means no more work from the master process.
            if chunk is None:
                break
            image_part = create_image(*chunk)
            comm.send((chunk, image_part), dest=0)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_command_line_args()
    width = args.width
    height = args.height
    max_iteration = args.max_iterations
    total_pixels = width * height

    if args.schedule == "static":
        static()
    else:
        dynamic()
