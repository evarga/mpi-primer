# Table of Contents
-  [Introduction](#introduction)
-  [Usage](#usage)
    - [Setup](#setup)
    - [Running the Application](#running-the-application)
      - [Static Scheduling](#static-scheduling)
      - [Dynamic Scheduling](#dynamic-scheduling)
-  [Conclusion](#conclusion)

# Introduction
This is a teaching material aimed to demonstrate the powerfulness of the [Single Program Multiple Data](https://www.geeksforgeeks.org/single-program-multiple-data-spmd-model/) (SPMD) paradigm 
with MPI[^1]. More specifically, this repo illustrates the foundational principles of distributed programming 
using a network of multicore/multiprocessor nodes. The following topics are covered in this unit:

- How the [Message Passing Interface](https://www.mpi-forum.org) (MPI) paradigm helps attain good performance by splitting data among parallel 
  processes potentially executing on different machines. Both static and dynamic scheduling are covered.
- The illustration of the [Scatter/Gather](https://mpi4py.readthedocs.io/en/stable/tutorial.html#collective-communication) collective communication pattern in MPI.
- The illustration of the [Send/Receive](https://mpi4py.readthedocs.io/en/stable/tutorial.html#point-to-point-communication) point-to-point communication pattern in MPI.
- What is a vectorized computation and how to do it in [NumPy](https://numpy.org).
- Why virtual environments are so important, and how to make one leveraging the standard Python 3+ toolset.
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
3. List the available packages to verify that everything is installed correctly:
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
To read the help message and learn what options are available run the following command:
```bash
mpiexec -n 1 python mpi-mandelbrot.py --help
```

### Static Scheduling
Here is the dump of the session producing a smaller 1000x1000 image using different number of processes with 
a default static scheduling policy (`--schedule=static`):
```
> time mpiexec -n 1 python mpi-mandelbrot.py 1000 1000
mpiexec -n 1 python mpi-mandelbrot.py 1000 1000  12.66s user 1.47s system 104% cpu 13.504 total
> time mpiexec -n 2 python mpi-mandelbrot.py 1000 1000                                         
mpiexec -n 2 python mpi-mandelbrot.py 1000 1000  14.36s user 0.96s system 202% cpu 7.556 total
> time mpiexec -n 6 python mpi-mandelbrot.py 1000 1000
mpiexec -n 6 python mpi-mandelbrot.py 1000 1000  29.37s user 1.88s system 566% cpu 5.514 total
```
The `time` command is used to measure the time it takes to run the program. The `mpiexec` command is used to run the 
program with a different number of processes. The `--output` option is used to specify the name of the output file. 
The first argument is the width of the image, and the second argument is the height of the image.

Notice that the time it takes to run the program decreases as the number of processes increases. This is because the
work is being distributed among the processes, and they are working in parallel. Nevertheless, the speedup is not 
linear when the number of processes is > 2 due to the overhead of communication between the processes, sequential stage of 
processing received parts by the master process, and imperfect load balancing.

The following two images show how work is distributed among the processes (each process is colored differently). In
static scheduling the work is evenly distributed among the processes. Nevertheless, this doesn't mean that the actual 
work done by each process will be the same.

**Note:** all images below are animated GIFs, so wait couple of seconds for a transition to happen from the base image to
the one depicting work distribution.

<kbd>![Mandelbrot_with_2_processes and static scheduling](images/mandelbrot-static-p2.gif)</kbd>

*Figure 1 - Work distribution among 2 processes with static scheduling.*

<kbd>![Mandelbrot with 6 processes and static scheduling](images/mandelbrot-static-p6.gif)</kbd>

*Figure 2 - Work distribution among 6 processes with static scheduling.*

Here is the dump of the session producing a larger 2000x2000 image using different number of processes with 
a default static scheduling policy:
```
> time mpiexec -n 1 python mpi-mandelbrot.py 2000 2000
mpiexec -n 1 python mpi-mandelbrot.py 2000 2000  58.38s user 11.62s system 100% cpu 1:09.75 total
> time mpiexec -n 2 python mpi-mandelbrot.py 2000 2000
mpiexec -n 2 python mpi-mandelbrot.py 2000 2000  62.69s user 9.36s system 199% cpu 36.119 total
> time mpiexec -n 6 python mpi-mandelbrot.py 2000 2000
mpiexec -n 6 python mpi-mandelbrot.py 2000 2000  145.42s user 9.73s system 576% cpu 26.920 total
```
Below, you have a case where, instead of increasing the data by x4, the amount of work per data chunk was increased by 
a factor x4. Observe that times are lower since less amount of data circulate around.
```
> time mpiexec -n 1 python mpi-mandelbrot.py --max_iterations 4000 1000 1000
mpiexec -n 1 python mpi-mandelbrot.py --max_iterations 4000 1000 1000  51.37s user 4.86s system 100% cpu 55.873 total
> time mpiexec -n 2 python mpi-mandelbrot.py --max_iterations 4000 1000 1000
mpiexec -n 2 python mpi-mandelbrot.py --max_iterations 4000 1000 1000  51.53s user 1.04s system 202% cpu 25.980 total
> time mpiexec -n 6 python mpi-mandelbrot.py --max_iterations 4000 1000 1000
mpiexec -n 6 python mpi-mandelbrot.py --max_iterations 4000 1000 1000  110.84s user 2.48s system 588% cpu 19.269 total
```

### Dynamic Scheduling
To implement dynamic scheduling, we need to change the way we distribute the work among the processes. Instead of 
dividing the total work into equal parts and assigning each part to a process at the beginning, 
we will divide the work into smaller chunks and assign each chunk to a process when it becomes available.

The following two images show how work is distributed among the processes (each process is colored differently). In
dynamic scheduling the work is not evenly distributed among the processes. They wait for the master process to send them
a new chunk of work when they are done with the previous one. The master process itself is also doing work when nothing
is ready from workers.

<kbd>![Mandelbrot_with_2_processes and dynamic scheduling](images/mandelbrot-dynamic-p2.gif)</kbd>

*Figure 3 - Work distribution among 2 processes with dynamic scheduling.*

<kbd>![Mandelbrot_with_6_processes and dynamic scheduling](images/mandelbrot-dynamic-p6.gif)</kbd>

*Figure 4 - Work distribution among 6 processes with dynamic scheduling.*

Here is the dump of the session producing a larger 2000x2000 image using different number of processes with 
a dynamic scheduling policy:
```
> time mpiexec -n 1 python mpi-mandelbrot.py --schedule=dynamic 2000 2000
mpiexec -n 1 python mpi-mandelbrot.py --schedule=dynamic 2000 2000  39.86s user 3.92s system 101% cpu 43.270 total
> time mpiexec -n 2 python mpi-mandelbrot.py --schedule=dynamic 2000 2000
mpiexec -n 2 python mpi-mandelbrot.py --schedule=dynamic 2000 2000  44.24s user 1.21s system 199% cpu 22.760 total
> time mpiexec -n 6 python mpi-mandelbrot.py --schedule=dynamic 2000 2000
mpiexec -n 6 python mpi-mandelbrot.py --schedule=dynamic 2000 2000  74.69s user 2.55s system 550% cpu 14.027 total
```
The times are lower than in the static scheduling case. This is especially evident when instead of increasing the 
amount of data we rise the number of iterations. Here is an example of a 1000x1000 image with 4000 iterations per pixel and 6
processes:
```
> time mpiexec -n 6  python mpi-mandelbrot.py --schedule=dynamic --max_iterations 4000 1000 1000
mpiexec -n 6 python mpi-mandelbrot.py --schedule=dynamic --max_iterations 400  53.19s user 2.95s system 574% cpu 9.774 total
```

# Conclusion
This project demonstrates the importance and usefulness of knowing ways to easily employ parallel and distributed programming concepts. 
Observe that you can easily scale the above examples to execute processes on different nodes. All this is 
completely handled by the underlying infrastructure. No need to touch the source code. MPI is a powerful tool for
distributed computing, and it is widely used in the scientific community. The beauty is that your code can be written 
as a sequential program with well-defined synchronization points.

It is very important to implement your code run by any worker process in efficient manner. In this project vectorized 
computation is employed thankfully to the NumPy library. Another popular hybrid parallel programming model is the 
combination of MPI and OpenMP[^2]. The former is used for distributed memory parallelism, and the latter is used for
shared memory parallelism.

Evidently, load balancing is of crucial importance to attain good performance. In this case study, dynamic scheduling
has turned out to be a better option, although this cannot be generalized. Sometimes a simple static scheduling
achieves better results, when evenly distributing a work is OK. For example, calculating a definite integral over some
range could be parallelized by splitting this range into equal subranges; no need for extra complexity and
overhead of dynamic scheduling.

[^1]: This project uses the [MPI for Python](https://mpi4py.readthedocs.io/en/stable/index.html) distribution.
[^2]: There is a separate [educational unit](https://github.com/evarga/openmp-primer) showcasing OpenMP.
