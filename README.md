# GPU Programming 

A transistor is a device that can increase the strength of an electronic signal or that can control the flow of the current. 

An integrated circuit (IC) is a mini electronic circuit. It combines transistors, resistors, capacitors, inductors into a single and small chip. 

We can control the software cost by redesigning the software less. But how we can avoid redesigning the software more ? 

Well, we can make the application scalable and portable. But what do we mean by scalability and portability ? Scalability is the ability of the application to handle large amounts of workloads by taking advantage of additional computing resources. Portability, on the other hand, is the ability for the application to run on multiple computing environments with minimal modifications.

So, how we can make the application scalable ? 

- More cores
- More threads per core
- More memory
- Faster interconnect

And how can we make the application portable ? 

- Supporting different instruction sets.
- Adapting to various processing units.
- Handling different memory architectures.

Heterogeneous parallel programming means developing software that can use multiple processing units simultaneously to take advantage of their unique strengths. But might be these processing units ?

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

The challenge in here is to design computer architectures and software development methods that can run older sequential programs efficiently without causing any slowdowns on new hardware and also allow new/highly parallel programs to take advantage of multiple cores and processors simultaneously for increased performance. 

### GPU vs CPU 

<img width="683" alt="image" src="https://github.com/user-attachments/assets/b5226712-6dc6-411a-859d-4b9d1c92fff2">

As we can see from above, the CPU has 4 large complex cores. These cores are designed for general purpose computing. Each of these cores has its own control unit. These control units are responsible from instruction decoding and execution management. They are crucial for managing the flow of instructions in sequential code We also see different levels of caches (L1, L2, and L3) that are used for quick data access. Lastly, there is a DRAM (Dynamic Random Access Memory) that serves as the main memory of the system. It stores the data/instructions that are actively used by the CPU. CPU is optimized for sequential code performance. 

In the GPU, however, we see smaller and higher number of cores (these are represented as green). These are used for parallel processing. Similarly, we see smaller and higher number of L1 caches and control units. Having a smaller L1 caches and control units minimizes the overhead for individual core control and maximizes space for processing units. Aside from that, we see a single and larger L2 cache that is shared by all cores and DRAM that stores data/instructions that are actively used by the GPU.

GPU can transfer data to/from its memory about 10 times faster than a typical multicore CPU thanks to:

- wider memory bus width compared to CPU
- specialized memory
- simpler memory access patterns compared to CPU
- parallel memory controllers
- optimized for throughput
- on-die memory interfaces
- relaxed memory model

This is an important point because GPUs need to feed data to many cores simultaneousy for parallel processing. That's why it should be able to transfer data to/from its memory very quickly.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/ee48dd71-fd18-4643-8ae9-fd18e2985398">

In the picture above, we see the key terms and concepts in computer hardware architecture.

- Chip: A chip is a integrated circuit that combines transistors, resistors, capacitors, and inductors in a single piece of material. These components are interconnected. Chips can perform many different tasks such as processing (e.g., CPU, GPU), memory storage (e.g., RAM, ROM chips), power management, etc. They can be categorized into:
  - Memory
  - Processor
    - Multicore
    - Single core  
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

In the picture below, we see the architectures of integrated and discrete GPU. 

<img width="536" alt="image" src="https://github.com/user-attachments/assets/3ab207cf-0e19-4c49-9999-e9d15bffe4fb">

**Note**: The northbridge is a chip that connects CPU, RAM, and GPU, and that manages data flow between these components.

Discrete GPU: Discrete GPUs have a dedicated hardware and memory. Therefore, they can work as more powerful processors and they can be designed in such a way that they are optimized for specific graph tasks. With discrete GPU, we can do more complex computations and obtain higher frame rates in applications that require many operations. 

Also, having a separate memory that is dedicated to graphics tasks (VRAM) allows faster data access and this is important point for handling large textures and complex 3D models efficiently. 

But the dedicated hardware and memory and additional complexity increase manufacturing costs, require more energy, and generate more heat that needs to be controlled.

Integrated CPU: The resources are shared between CPU and GPU. The important factor in here is power efficiency. Low energy is a key design goal and this makes integrated CPU ideal for mobile devices.

Combining CPU and GPU reduces the overall chip count and makes manufacturing more simple. 

Because the memory is shared between CPU and GPU, we observe performance bottlencek for graphic intensive tasks in addition to more flexible memory allocation. 

Where we can use GPU ? GPU is typically suitable if:

- The application is computation intensive and requires large amounts of data that needs to be processed because GPUs are designed to handle large numbers of calculations in parallel. During this process, data is transferred from/to GPU and if the application is not computationally intensive, it would not be worth to use GPU.
- The application has many independent computations because GPUs can perform many calculations simultaneously but only if these calculations are independent from each other.
- The application has many similar computations because GPUs are efficient the most when executing the same instructions across multiple data points.

Applications that don't meet one or more of these criteria would better to be executed by CPU. 

<img width="400" alt="image" src="https://github.com/user-attachments/assets/e69e3e55-f435-49ee-b203-6a36b69b3162">

<img width="400" alt="image" src="https://github.com/user-attachments/assets/cdb39e77-3264-47f4-835f-640d50f49395">

At the top of the diagram we see host. The host is typically CPU. And input assembler prepare and send data to the GPU. Thread execution manager manages the distribution and execution of threads across the GPU.

The green squares represent streaming processors. A streaming processor is also called CUDA core. Each streaming processor (each green square) is a processing unit that is capable of executing arithmetic and logical operations. These streaming processors are designed to work in parallel and perform the same operation on different data simultaneously.

A streaming multiprocessor is a processing unit that contains multiple smaller streaming processors. It includes shared memory/L1 cache, texture units, load/store units, and special function units (for complex amth operations). 

Parallel data cache serves as a memory and it stores frequently accessed data. Streaming processors can access data in these caches simultaneously.

In 3D graphics, texture is an image that is applied to the mathematical representation of a 3D object in a digital space. 

We apply these images because 3D objects are just shapes and we want to add color, patterns, and other details. It is more efficient to apply a detailed image to a 3D object compared to modeling every single detail in 3D object. 

Load/store units are specialized circuits that are part of the GPUs. They handle the operations of reading data from global memory into registers/processing units and writing data from registers/processing units back to memory. 

Load/store units in the GPU are designed in such a way that they combine multiple memory requests into fewer but larger transactions for efficiency. Multiple load/store units operate in parallel to support the many concurrent threads in a GPU. Efficent load/store operations are crucial for tasks such as accessing large datasets. These load/store operations play a key role in matrix multiplications in neural networks. 





















