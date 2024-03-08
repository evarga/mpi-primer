# Table of Contents
-  [Introduction](#introduction)
-  [Usage](#usage)
    - [Setup](#setup)
    - [Running the Application](#running-the-application)
-  [Conclusion](#conclusion)

# Introduction
This is a teaching material aimed to demonstrate the powerfulness of the [Single Program Multiple Data](https://www.geeksforgeeks.org/single-program-multiple-data-spmd-model/) (SPMD) paradigm 
with MPI[^1]. More specifically, this repo illustrates the foundational principles of distributed programming 
using a network of multicore/multiprocessor nodes. The following topics are covered in this unit:

- How the [Message Passing Interface](https://www.mpi-forum.org) (MPI) paradigm helps attain good performance by splitting data among parallel processes potentially executing on different machines.
- The illustration of the [Scatter/Gather](https://mpi4py.readthedocs.io/en/stable/tutorial.html#collective-communication) collective communication pattern in MPI.
- What is a vectorized computation and how to do it in [NumPy](https://numpy.org).
- Why virtual environments are so important, and to make one leveraging the standard Python 3+ toolset.
- An example of a [fractal](https://en.wikipedia.org/wiki/Fractal) image called the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set).
- How to produce an animated GIF in Python using the [ImageIO](https://imageio.readthedocs.io/en/stable/) library.
- How to parse command line arguments and provide a help system at the command line.

# Usage
It is assumed that all commands below will be executed from the project's *root* folder as well as that this repo 
was cloned from GitHub and is available on your machine. Furthermore, it is assumed that you have Python 3.10+ 
installed on your machine and is invoked via `python` as well as it's package manager as `pip`. 
If this is not the case, then you will need to adjust the instructions below accordingly. Finally, Windows users are
expected to use the [Cygwin](https://www.cygwin.com) environment.

## Setup
Follow the steps below to set up the virtual environment:

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   As a sanity check you may want to run `echo $VIRTUAL_ENV` to see if the environment is activated.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. List the installed packages to verify that everything is installed correctly:
   ```bash
   pip list
   ```
   This should produce the following output:
   ```
   Package Version
   ------- -------
   imageio 2.34.0
   mpi4py  3.1.5
   numpy   1.26.4
   pillow  10.2.0
   pip     24.0
   ```
4. Deactivate the virtual environment once you are done running this project:
   ```bash
   deactivate
   ```
The advantage of using a virtual environment is that it allows you to work on multiple projects with different 
dependencies without them interfering with each other. It also makes it easy to share your project with others, 
as they can create the same environment on their machine.

## Running the Application
To read the help message, run the following command:
```bash
mpiexec -n 1 python mpi-mandelbrot.py --help
```
Here is the dump of the session producing a smaller 1000x1000 image using different number of processes:
```
> time mpiexec -n 1 python mpi-mandelbrot.py --output small1.gif 1000 1000
mpiexec -n 1 python mpi-mandelbrot.py --output small1.gif 1000 1000  13.49s user 1.65s system 102% cpu 14.816 total
> time mpiexec -n 2 python mpi-mandelbrot.py --output small2.gif 1000 1000
mpiexec -n 2 python mpi-mandelbrot.py --output small2.gif 1000 1000  15.25s user 0.94s system 208% cpu 7.782 total
> time mpiexec -n 6 python mpi-mandelbrot.py --output small6.gif 1000 1000
mpiexec -n 6 python mpi-mandelbrot.py --output small6.gif 1000 1000  32.60s user 2.01s system 532% cpu 6.496 total
```
The `time` command is used to measure the time it takes to run the program. The `mpiexec` command is used to run the 
program with a different number of processes. The `--output` option is used to specify the name of the output file. 
The first argument is the width of the image, and the second argument is the height of the image.

Notice that the time it takes to run the program decreases as the number of processes increases. This is because the
work is being distributed among the processes, and they are working in parallel. Nevertheless, the speedup is not 
linear due to the overhead of communication between the processes, sequential stage of processing received parts by the 
master process, and imperfect load balancing.

The following two images show how work is distributed among the processes (each assignment is colored differently). 
The first image shows the work being distributed among two processes, and the second image shows the work being distributed among six processes. The source code of the program is available in the `mpi-mandelbrot.py` file. It contains detailed explanations of how the program works and why there is a greater imbalance with 6 processes.

<kbd>![Mandelbrot_with_2_processes](images/mandelbrot-p2.gif)</kbd>
<kbd>![Mandelbrot_with_6_processes](images/mandelbrot-p6.gif)</kbd>

Here is the dump of the session producing a larger 2000x2000 image using different number of processes:
```
> time mpiexec -n 1 python mpi-mandelbrot.py --output large1.gif 2000 2000
mpiexec -n 1 python mpi-mandelbrot.py --output large1.gif 2000 2000  58.68s user 11.48s system 100% cpu 1:09.60 total
> time mpiexec -n 2 python mpi-mandelbrot.py --output large2.gif 2000 2000
mpiexec -n 2 python mpi-mandelbrot.py --output large2.gif 2000 2000  66.84s user 10.09s system 198% cpu 38.845 total
> time mpiexec -n 6 python mpi-mandelbrot.py --output large6.gif 2000 2000
mpiexec -n 6 python mpi-mandelbrot.py --output large6.gif 2000 2000  157.65s user 11.12s system 547% cpu 30.824 total
```

# Conclusion
This project demonstrates the importance and usefulness of knowing ways to easily employ parallel and distributed programming concepts. Observe that you can easily scale the above examples to execute processes on different nodes. All this is 
completely handled by the underlying infrastructure. No need to touch the source code. MPI is a powerful tool for
distributed computing, and it is widely used in the scientific community. The beauty is that your code can be written 
as a sequential program with well-defined synchronization points.

It is very important to implement your code run by any worker process in efficient manner. In this project vectorized 
computation is employed thankfully to the NumPy library. Another popular hybrid parallel programming model is the 
combination of MPI and OpenMP[^2]. The former is used for distributed memory parallelism, and the latter is used for
shared memory parallelism.

[^1]: This project uses the [MPI for Python](https://mpi4py.readthedocs.io/en/stable/index.html) distribution.
[^2]: There is a separate [educational unit](https://github.com/evarga/openmp-primer) showcasing OpenMP.