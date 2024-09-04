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







