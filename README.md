# GPU Programming 

## Lecture 1

A transistor is a device that can increase the strength of an electronic signal or that can control the flow of the current. We can't pack more features inside the GPU unless we have more transistors. If we can't put more transistors into the GPU, we need another way to improve performance. So, instead of using one GPU, we may use multiple GPUs and make multiple GPUs to interact with each other. But the issue is that if you increase the number of GPUs beyond some number, the communication among them will kill the performance you are getting. 

An integrated circuit (IC) is a mini electronic circuit. It combines transistors, resistors, capacitors, inductors into a single and small chip. 

We can control the software cost by redesigning the software less. But how we can avoid redesigning the software ? 

Well, we can make the application scalable and portable. But what do we mean by scalability and portability ? Scalability is the ability of the application to handle large amounts of workloads by taking advantage of additional computing resources. The key question is: if I take the code written for GPU and run it on another GPU with bigger capabilities, will I see performance improvement ? 

Portability, on the other hand, is the ability for the application to run on multiple computing environments with minimal modifications. If I am writing a code for the GPU, the code must continue working when a newer GPU comes with minimal change. 

So, how we can make the application scalable ? 

- More cores
- More threads per core
- More memory
- Faster interconnect

And how can we make the application portable ? 

- Supporting different instruction sets.
- Adapting to various processing units.
- Handling different memory architectures.

Heterogeneous parallel programming, on the other hand, means developing software that can use multiple processing units simultaneously to take advantage of their unique strengths. The examples of processing units can be seen in below:

- CPUs (Central Processing Units)
- GPUs (Graphics Processing Units)
- FPGAs (Field-Programmable Gate Arrays)
- DSPs (Digital Signal Sensors)
- AI Accelerators (e.g., Tensor Processing Units)

<img width="678" alt="image" src="https://github.com/user-attachments/assets/d0ac6f33-d0cb-40cf-925b-466eff31d22b">

When we deal with parallel programming, we transition from single-core to multicore and parallel programming architectures. That's why:

- we want to maintain the execution speed of programs that are designed to run on a single core (these are also called sequential programs).
- we want to maintain the execution speed of programs in which large operations run concurrently.
- we want to increase the throughput of programs that are designed to run on multiple cores/processors simultaneously (parallel programs)

The challenge in here is to design computer architectures and software development methods that can run older sequential programs efficiently without causing any slowdowns on new hardware and also that allow new/highly parallel programs to take advantage of multiple cores and processors simultaneously for increased performance. 

Some languages made parallel programming easier but none has made it as fast, efficient, and flexible as traditional sequential programming. 

### GPU vs CPU 

<img width="683" alt="image" src="https://github.com/user-attachments/assets/b5226712-6dc6-411a-859d-4b9d1c92fff2">

As we can see from above, the CPU has 4 large complex cores. These cores are designed for general purpose computing. Each of these cores has its own control unit. These control units are responsible from instruction decoding and execution management. They are crucial for managing the flow of instructions in sequential code. In other words, traditional multicore CPU is designed for sequential instruction. We also see different levels of caches (L1, L2, and L3) that are used for quick data access. Lastly, there is a DRAM (Dynamic Random Access Memory) that serves as the main memory of the system. It stores the data/instructions that are actively used by the CPU. CPU is optimized for sequential code performance. 

In the GPU, however, we see smaller and higher number of cores (these are represented as green). These are used for parallel processing. Similarly, we see smaller and higher number of L1 caches and control units. Having a smaller L1 caches and control units minimizes the overhead for individual core control and maximizes space for execution units (cores). Aside from that, we see a single and larger L2 cache that is shared by all cores and DRAM that stores data/instructions that are actively used by the GPU.

When you design DRAM, you can optimize it either for latency or for bandwidth (getting a large amount of data at once). GPU is optimized for bandwidth. It is a little bit slower than CPU but once the process is done, you can get a huge amount of data at once. The memory for multicore CPU, on the other hand, is optimized for latency.

Designing GPU for a high bandwidth is an important point because GPUs need to feed data to many cores simultaneousy for parallel processing. That's why it should be able to transfer large amount of data to/from its memory very quickly.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/ee48dd71-fd18-4643-8ae9-fd18e2985398">

In the picture above, we see the key terms and concepts in computer hardware architecture.

- Chip: A chip is a integrated circuit that combines transistors, resistors, capacitors, and inductors in a single piece of material. These components are interconnected. Chips can perform many different tasks such as processing (e.g., CPU, GPU), memory storage (e.g., RAM, ROM chips), power management, etc. They can be categorized into:
  - Memory
  - Processor
    - Multicore processor
    - Single core processor
  - Accelerator
    - GPU
    - TPU
    - FPGA
   
So, the question is: how do we choose a processor for our application ? 

The key factors that we consider when choosing a processor for our application are: 

- Performance: How fast and quick we want our tasks to be executed ?
- Very large installation base: Choosing a processor that is widely used in the industry because it provides better software compatibility and support.
- Practical form-factor (physical size and shape of processor) and easy accessibility (how easy it is to purchase and and how easily it can be integrated/replaced with common motherboards and system design)
- Support for IEEE754 floating point standard: IEEE754 defines how floating point numbers should be represented/computed. To calculate accurate floating points consistently and precisely, this support is needed.

### Integrated GPU vs Discrete GPU

In the picture below, we see the architectures of integrated and discrete GPU. 

<img width="536" alt="image" src="https://github.com/user-attachments/assets/3ab207cf-0e19-4c49-9999-e9d15bffe4fb">

**Note**: _The northbridge is a chip that connects CPU, RAM, and GPU, and that manages data flow between these components._

**Discrete GPU:** Discrete GPUs have a dedicated hardware and memory. You can plug a discrete GPU into the PCI express port.Therefore, they can work as more powerful processors and they can be designed in such a way that they are optimized for specific graph tasks. With discrete GPU, we can do more complex computations and obtain higher frame rates in applications that require many operations. 

Also, having a separate memory that is dedicated to graphics tasks (VRAM) allows faster data access and this is important point for handling large textures and complex 3D models efficiently. 

But the dedicated hardware and memory and additional complexity increase manufacturing costs, require more energy, and generate more heat that needs to be controlled.

**Integrated GPU:** The GPU is embedded inside the multicore CPU. The resources are shared between CPU and GPU and the advantage of this architecture is that the communication between the multicore CPU and GPU is very fast. Also, combining CPU and GPU reduces the overall chip count and makes manufacturing more simple. The disadvantage is that we observe performance bottlencek for graphic intensive tasks in addition to more flexible memory allocation because the memory is shared between CPU and GPU. Also, this is a very weak GPU because we don't have space inside the chip to have a fully fledged GPUs. So, we have a lower performance compared to discrete GPU. 

The important factor in this architecture is power efficiency. Low energy is a key design goal in integrated GPU and this makes this architecture ideal for mobile devices.

### When GPU Should be Used ? 

GPU is typically suitable if:

- The application is computation intensive and requires large amounts of data that needs to be processed because GPUs are designed to do only computation and to handle large numbers of calculations in parallel. During this process, data is transferred from/to GPU and if the application is not computationally intensive, it would not be worth to use GPU.
- The application has many independent computations because GPUs can perform many calculations simultaneously but only if these calculations are independent from each other. If a computation is waiting for the result of another computation, this means that the parallelism is reduced. Moving data from the system memory to GPU memory is pretty expensive. That will cancel off what you paid for moving the data from system memory to GPU. 
- The application has many similar computations because GPUs are efficient the most when executing the same instructions across multiple data points. The reason is that GPUs are designed in such a way that the same piece of code will be executed on different pieces of data.
- The size of the problem is big. If you want to do matrix multiplication for very small matrices, it is better to do it sequentially because you are paying a lot to move the data from system memory to GPU and this won't be worth for this kind of small task.

Applications that don't meet one or more of these criteria would better to be executed by CPU. 

### GPU Architecture

<img width="400" alt="image" src="https://github.com/user-attachments/assets/e69e3e55-f435-49ee-b203-6a36b69b3162">

<img width="400" alt="image" src="https://github.com/user-attachments/assets/cdb39e77-3264-47f4-835f-640d50f49395">

At the top of the diagram we see host. The host is typically CPU. And input assembler prepare and send data to the GPU. Thread execution manager manages the distribution and execution of threads across the GPU.

The green squares represent streaming processors. These are also called execution units or CUDA cores (even though they are actually not like traditional CPU cores). Each streaming processor (each green square) is an execution unit that is capable of executing arithmetic and logical operations. These streaming processors are designed to work in parallel and perform the same operation on different data simultaneously.

A streaming multiprocessor is a processing unit that contains multiple smaller streaming processors. It includes shared memory/L1 cache, texture units, load/store units, and special function units (for complex math operations). 

Parallel data cache serves as a memory and it stores frequently accessed data. Streaming processors can access data in these caches simultaneously.

In 3D graphics, texture is an image that is applied to the mathematical representation of a 3D object in a digital space. 

We apply these images because 3D objects are just shapes and we want to add color, patterns, and other details. It is more efficient to apply a detailed image to a 3D object compared to modeling every single detail in 3D object. 

Load/store units are specialized circuits that are part of the GPUs. They handle the operations of reading data from global memory into registers/processing units and writing data from registers/processing units back to memory. 

Load/store units in the GPU are designed in such a way that they combine multiple memory requests into fewer but larger transactions for efficiency. Multiple load/store units operate in parallel to support the many concurrent threads in a GPU. Efficent load/store operations are crucial for tasks such as accessing large datasets. These load/store operations play a key role in matrix multiplications in neural networks. 

The CUDA cores need to communiate with each other in some rare scenarios. At some point, they may need to share some data and have some communication among them. But note that we don't mean fully connecting the cores with each other. This would not be logical because we don't have space, fully connecting them increases the amount of generated heat. So, this would be prohibitive. 

But we also don't want to disconnect them completely since sometimes they may need to communicate with each other. And if all the CUDA cores are completely disconnected, they will have to send/receive data through memory which is quite slow and this will have negative effects on the performance. 

Because of these reasons, we see a GPU design in which a group of execution units are put into group together so that the execution units in each group can communicate with other execution units in the same group very quickly. Within the same group, the execution units can send/receive data very quickly without having to use memory. But if an execution unit wants to send data to the execution unit in a different group, this has to be done through GPU memory (not the system memory) which is expensive. 

One note is that although the execution units (green squares) are called streaming processors or CUDA cores, they are neither processors nor cores that we know from the CPU. They are just execution units that are responsible from executing instructions. The execution units in GPU are a little bit more sophisticated than the eexcution units in CPU because they also write the results back to the registers or memory. Every one of the CUDA cores have its own commit space.

A warp is a group of threads that are executed together in parallel on the GPU. A warp typically consits of 32 threads. All threads in a warp (a group of 32 threads), execute the same instruction at the same time but on a different data. This is part of SIMT (Single Instruction, Multiple Thread). Through this way, GPUs can process large amounts of data quickly. GPU's scheduler works with warps instead of working with individual threads. The warps are scheduled to run on the available processing units. 

For all 32 threads in a warp, the same instruction is fetched once, the instruction is decoded once and the decoded instruction is broadcast to all threads in a warp. In other words, all the threads in a warp share the same front-end (the process of fetch, decode, etc.). This is an important feature because it helps us to save more space for the execution units. So, instead of having a front-end for every one of several thousands of CUDA cores, we allow a group of them to share the same front end.

Also, one additional note is that two streaming multiprocessors may look like they are grouped together but this is done so that each group of streaming multiprocessor share the same texture memory (which is used for graphics application). The reason why we want more than one streaming multiprocessors to share the same texture memory is to save more space for execution units.

### Amdahl's Law

One key point is that we should understand the limitations of optimization and parallelization in computing. To achieve significant speedups, we should optimize/parallelize as much of the program as possible and avoid focusing only one part. Because the overall speedup of a system is limited by the portion of the system that cannot be improved.

To predict the maximum speedup for a system, we can use this equation `Execution Time After Improvement = Execution Time Unaffected + Execution Time Affected / Amount of Improvement`

For instance, if there is a program that runs in 100 seconds, 80 seconds of this time is spent for multiplication process and if we want to make this program run 4 times faster: 

- Total execution time = 100 seconds
- Time spent on multiplication = 80 seconds (Execution Time Affected)
- Time spent on other operations = 20 seconds (Execution Time Unaffected)
- Target execution time (Execution Time After Improvement) = 100 seconds / 4 = 25 seconds

25 seconds = 20 seconds + 80 / Amount of Improvement $->$ Amount of Improvement = 16. 

So multiplication must be 16 times faster if we want to make this program run 4 times faster. 

If we want to make this program run 5 times faster, this would not be possible because 

20 seconds ≠ 20 seconds + 80 / Amount of Improvement. 

In summary improvement in the application speed depends on the portion that is parallelized. 

### How to Decide CPU vs GPU
Lastly, how do we decide if we should use CPU or GPU ? 

- CPUs are better for sequential code or tasks with low/heterogeneous parallelism because
  - they have branch prediction, out-of-order execution, and some other techniques that are used to maximize the speed of individual instructions and sequential code,
  - the memory hierarchy in the CPU is optimized for the access patterns of sequential code,
  - CPUs have advanced mechanisms that allow them to handle branching, conditional execution, and other control flow constrcuts that are common in sequential code.
 
- GPUs are better for parallel computation where the total amount of work that is completed over time (throughput) is the priority.

The key points that should be empahsized: 

- Not all parts of a program can be easily parallelized. Some tasks have dependencies or require sequential execution. This limits the amount of work that can be put into the GPU. If only small amount of code is parallelizable, the overall speedup of using GPU on running this code may be limited.
- CPU and GPU need to communicate with each other whenever there are tasks that can be accelerated by putting some of the work to the GPUs parallel processing units. For instance, let's assume that there is a 3D scene or high resolution graphics and CPU wants to send the data that needs to be rendered and rendering isntructions to GPU. After GPU performs the rendering instructions on the data, it sends the output back to the CPU because CPU may need to perform some additional tasks on this output. Also, the CPU is responsible from managing the system's main memory. When GPU finishes its operations, it may need to send the output to main memory so that other parts of the system can access them and CPU handles this data transfer from GPU to memory. Even if GPU may want to send the output to another parts of the computer such as display without going through the CPU, CPU is still involved in setting up the display and managing the overall process. CPU tells the GPU where to render the output and where to display it. The communication between CPU and GPU takes additional time. If the amount of data that is transferred from/to GPU to/from CPU is large or if CPU and GPU need to communicate with each other very frequently, this may reduce the performance and should be taken into account during the system design.
- As we mentioned before, GPUs have their own dedicated memory. This memory is called Video RAM (VRAM). It is basically used to store data and intermediate results during the computation. The data rate at which data can be read/written from/to VRAM is called memory bandwidth. Memory bandwidth is determined based on several factors that are related to the memory system's design and configuration. These factors are like these:
  - Memory clock speed
  - Memory bus width
  - Memory type
  - Number of memory channels
  - Memory access patterns
  - Memory latency
  - Cache usage
  - Resource contention
  - Power and thermal constraints

So if the memory bandwidth is saturated, this means that GPU is unable to read/write data as quickly as it can process it and this causes to bottleneck in performance. This problem can be solved with using data compression techniques, GPUs with higher memory bandwidth or optimizing memory access patterns. 

#### Lecture 1 Summary

- Communication and memory access are very expensive
- Sometimes recomputation is less expensive than sending data from multicore to GPU
- We need to optimize communication and memory - not computation
- Accelerator is a chip that is designed to execute speical type of application very fast and very efficient. But it is pretty bad for other tasks. GPU is one of these accelerators. It can execute other applications but it is pretty bad.
- The cache in the multicore CPU takes 70% of the space. This is not what we want in GPU.
- Relationship between clock speed and performance.
- Pim (processing in memory)
- Independency is important for GPU but it is very rare to see 100% independence. There is still some dependency to some extent.
- CUDA core are just execution units. They just execute (Taken picture in Photos)
- Each group shares the same texture to save more space for execution.
- System memory is optimized for latency. GPU memory is optimized for bandwidth.

## Recitation 1 

Theoretically, GPUs can perform any computation that a traditional CPU can. GPUs are designed to handle many simple calculations simultaneously. For tasks that can be broken down into many independent and similar operations, GPUs can significantly outperform CPUs. Also GPUs have thousands of cores that can execute threads in parallel. 

Although GPUs provide very fast computations for parallel computations, they are not good for general purpose computation which has a lot of sequential computations because of the reaosons below: 

- Low clock speed. Clock speed represents the number of cycles a processor can execute per second. It is a measure of how quickly a processor can perform individual operations. Cycle, on the other hand, means a fundamental unit of operation in digital electronics. During a cycle, a processor performs a basic operation or part of an operation. An example of this might be fetching an instruction, decoding it, executing it, or accessing memory. A clock speed of 3 GHz, for instance, means that the processor completes 3 billion cycles per second. GPUs have a low clock speed because they are designed for massive parallelism with thousansd of cores that are optimized for performing many simple operations simltaneously rather than complex sequential operations. Also, lower clock speeds consume less power per core. With thousands of cores, running at high clock speeds would cause excessive power consumption and heat generation. In addition, the priority of the GPUs is overall throughput over the speed of individual operations. T
- GPU cores are simple. They are designed to be specialized and to focus on performing specific types of computations efficiently. They are optimized for parallel processing of relatively simple and repetitive tasks.
- GPUs have a more limited and specialized instruction set compared to CPUs. The instruction set of a processor is the set of basic operations it can perform.
- GPUs have limited branch prediction. (Branch prediction unit is a special type of hardware CPU has. It guesses the correct path of the control flow statements with >90% accuracy. And this results in a faster execution for sequential operations).



## Lecture 2

Before GPUs, transformations were done on CPU. 3D objects are represented by vertices (points in 3D space). Vertices are represented by their X, Y, and Z coordinates and they are connected to form polygons (a figure with at least three straight sides and angles). The 3D model data that is defined by vertices is converted into 2D pixels on a screen. Before GPUs, all the operations that are needed to convert the 3D coordinates to 2D screen positions were performed by the CPU. This was not efficient because CPU had to calculate color and properties of each pixel one at a time sequentially. 

<img width="542" alt="image" src="https://github.com/user-attachments/assets/96fc38fb-ff2a-45b0-9fad-f0fb7f1744cd">

In computer graphics, each figure is basically made by using triangles. Triangle is made from 3 points (numbers or vertices in other words) and edges that connect these 3 vertices. 

In the picture above, the host is the main processor of the computer. It sends commands/data to the GPU. Host interface is an interface between the CPU and GPU. It handles communication and data transfer. Fixed communication pipeline means that the system is not editable. You buy the GPU as it is and you cannot change it. As part of the GPU pipeline:

- **Vertex control** receives vertex data, converts it into a form that hardware understands, and stores it in the vertex cache. 
- **VS/T & L (Vertex Shading/Transform & Lighting)** applies transformations and lighting calculations to each vertex.
- **Vertex cache** stores processed vertex data for reuse.
- During the **triangle setup**, triangles are prepared by using vertices for rasterization. (Rasterization means converting trianges into pixels).
- During **rasterization**, triangles are converted into pixels and the output is called raster. During the rasterization, which pixel will fall into which triangle is determined (we determine whether the pixel will be at the edge or between the two triangles, etc.) For each pixel, per-pixel values are interpolated from vertices. 
- **Shader** applies color and texture to pixels. It determines the final color of each pixel.
- **ROP (Raster Operations)** performs final operations like depth testing and blending. It performs color raster operations that blend the color of overlapping objects for transparency and antialiasing. 
- **FBI (Frame Buffer Interface)** manages memory reads/writes.

<img width="489" alt="image" src="https://github.com/user-attachments/assets/6ec0256c-4778-4653-a670-706863b42f4e">





