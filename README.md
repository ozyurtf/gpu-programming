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

For all 32 threads in a warp, the same instruction is fetched once, the instruction is decoded once and the decoded instruction is broadcast to all threads in a warp. In other words, all the threads in a warp share the same front-end (the process of fetch, decode, etc.). 

Fetch units and decoder units are collections of circuits and logic elements. They are made up from transistors and they work together to perform their respective functions. The fetch unit typically consists of program counter (PC), memory access logic, and prefetch buffer. Decoder unit consists of instruction register, decoding logic, and microcode ROM. These fetch units and decoder units are physically implemented on the CPU die. 

Having the same front-end for all 32 threads in a warp reduces dependencies between threads because if each thread would manage its own fetch-decode-execute pipeline, some threads might have to wait for others to complete. This would bring more dependencies and reduce parallelism. In addition, if each thread would manage its own front-end, it would result in multiple requests to the instruction cache or memory, and this would increase memory traffic significantly.

Aside from this, GPUs have a fixed die area. This die area is divided into computational units (SPs), control logic (which includes fetch and decode units), caches, memory interfaces, etc. By sharing the same front-end, less area is used for control logic. This allows more space for execution units and results in higher throughput.

Lastly, if all threads share the same front-end, they will be synchronized. Therefore, the system won't need to manage synchronization between these threads and coordination between different parts of the GPU will be simpler/easier to handle.

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

### Heterogeneous Parallel Computing

Multi-core design refers to the design strategy where a chip has multiple independent processing cores. Each of these cores is capable of executing its own threads simultaneously. The goal of multicore is to increase the number of threads (tasks) that can be run sycnhronously but with a focus on high performance for individual tasks that require sequential operations. 

Many-thread design refers to the ability to run thousands of threads in parallel. The cores in the GPU are simpler and smaller compared to the CPU but there are much more of them.

GPUs are evolved to handle parallel tasks such as rendering pixels on a screen and processing each pixel independently of others. This is a highly parallelizable task. Therefore, GPUs were designed to handle many threads simultaneously to maximize throughput.

The term many-thread is associated with GPUs because GPUs are built to handle workloads that involve thousands or even millions of threads in parallel. This allows them to be very good at handling highly parallel tasks such as rendering graphics, deep learning, and scientific computations. 

CPUs focus on running fewer tasks at the same time. Each of these tasks is given more resources compared to GPU (e.g., larger cache, complex control logic) ti maximize performance for sequential tasks. 

In GPUs, each thread doesn't require the same complex control or execution logic that a CPU thread does.

### Parallelizable Codes 

```
// Initialize two vectors of the same size
std::vector<int> vec1 = {1, 2, 3, 4, 5};
std::vector<int> vec2 = {6, 7, 8, 9, 10};

// Initialize a vector to store the result
std::vector<int> result(vec1.size());

// Perform element-wise addition using a for loop
for (size_t i = 0; i < vec1.size(); ++i) {
  result[i] = vec1[i] + vec2[i];
}
```
The code that is shown above is parallelizable. Each GPU core can compute result[i] value independently. 

```
std::vector<std::vector<int>> A = {
  {1, 2, 3},
  {4, 5, 6}
};

std::vector<std::vector<int>> B = {
  {7, 8},
  {9, 10},
  {11, 12}
};

// Matrix C to store the result, size is A's rows x B's columns
std::vector<std::vector<int>> C(A.size(), std::vector<int>(B[0].size(), 0));

// Matrix multiplication
for (size_t i = 0; i < A.size(); ++i) {
  for (size_t j = 0; j < B[0].size(); ++j) {
    for (size_t k = 0; k < B.size(); ++k) {
      C[i][j] += A[i][k] * B[k][j];
    }
  }
}
```

The code that is shown above is parallelizable as well. Each GPU core can read  ith row of A, jth column of B, take the dot product of this row-column pair to compute C[i][j] and write the result. 

```
// Define the array (vector)
std::vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8};

int sum = 0;
// Calculate the sum of elements in the array

for (size_t i = 0; i < arr.size(); ++i) {
  sum += arr[i];
}
```

This code can be parallelized but this won't be as easy as the vector addition or matrix multiplication example because in this code, each iteration depends on the result of the previous operation (sum += arr[i]). This dependency creates a race operation. However, this code can still be parallelized using parallel reduction. 

We can basically split the array into smaller chunks. Each of these chunks can be processed by a different GPU thread and each thread can compute a partial sum for its chunk. These partial sums can then be combined in a tree like structure. This process continues until the final sum is computed.

For instance, if we have the array of [1, 2, 3, 4, 5, 6, 7, 8]

Step 1: Split into chunks
Assume we have 4 GPU threads available. We'll split the array into 4 chunks:

Thread 1: [1, 2]
Thread 2: [3, 4]
Thread 3: [5, 6]
Thread 4: [7, 8]

Step 2: Compute partial sums
Each thread computes the sum of its chunk:

Thread 1: 1 + 2 = 3
Thread 2: 3 + 4 = 7
Thread 3: 5 + 6 = 11
Thread 4: 7 + 8 = 15

Step 3: Tree-like reduction
Now we have [3, 7, 11, 15]. We continue summing in parallel, reducing the number of active threads in each step:
First reduction:

Thread 1: 3 + 7 = 10
Thread 2: 11 + 15 = 26

Second reduction:

Thread 1: 10 + 26 = 36

Final result: 36

The tree-like structure looks like this:

```
        36
      /   \
    10     26
   /  \   /  \
  3   7  11  15
 / \ / \ / \ / \
1 2 3  4 5 6 7  8

```

```
struct Node {
  int data; // Data part of the node
  Node* next; // Pointer to the next node
};

// Function to traverse and update
void traverse(Node* head) {
  Node* current = head;
  while (current != nullptr) {
    current->data = (current->data %2) ? 0 : current->data;
    current = current->next;
  }
  std::cout << std::endl;
}
```

The code shown above is not suitable for parallelizing. Efficient parallelization requires dividing the work evenly among processors or threads. In the example above, we don't know the size of the linked list. So, this lack of size information makes it difficult to divide the work among parallel processors efficiently.

Also, each node in the linked list points to the next node. We can only find the location of the next node after accessing the current node. This sequential dependency prevents us from accessing to nodes in parallel. That's why it is not suitable for parallelizing. 

```
void bubbleSort(std::vector<int>& arr) {
  int n = arr.size();
  for (int i = 0; i < n - 1; ++i) {
    // Last i elements are already sorted
    for (int j = 0; j < n - i - 1; ++j) {
      // Swap if the element is greater than the next element
      if (arr[j] > arr[j + 1]) {
        std::swap(arr[j], arr[j + 1]);
      }
    }
  }
}
```

Bubble sort is a comparison based algorithm and it repeatedly steps through the list, compares adjacent elements, and swaps them if they are in wrong order. Having frequent, unpredictable swaps between adjacent elements results in data dependencies that are difficult to parallelize efficiently. The random nature of swaps can lead to uneven workloads across parallel processors and frequent memory conflicts.

```
int binarySearch(const std::vector<int>& arr, int left, int right, int target) {
  while (left <= right) {
    int mid = left + (right - left) / 2;

    // Check if target is at mid
    if (arr[mid] == target)
      return mid;

    if (arr[mid] < target)
      left = mid + 1;

    else
      right = mid - 1;
  }

  // Return -1 if target is not present in the array
  return -1;
}
```

The binary search algorithm can be parallelized. Let's say we have a sorted array of 16 elements and we're searching for the value 42:

Array: [2, 5, 8, 12, 16, 23, 28, 31, 37, 42, 46, 51, 55, 60, 64, 70]
Target: 42

Step 1: Initial Division
We'll divide this into 4 segments, each handled by a different thread:
Thread 1: [2, 5, 8, 12]
Thread 2: [16, 23, 28, 31]
Thread 3: [37, 42, 46, 51]
Thread 4: [55, 60, 64, 70]

Step 2: Parallel Local Search
Each thread performs a binary search on its segment:
Thread 1: 42 > 12, not found
Thread 2: 42 > 31, not found
Thread 3: 42 found at local index 1
Thread 4: 42 < 55, not found

Step 3: Combine Results
We now know that 42 is in the third segment. We can discard the other segments.

Step 4: Refine Search (if needed)
In this case, we've found the exact location. But let's say we didn't find it exactly and needed to refine further:
New array to search: [37, 42, 46, 51]

We could divide this again among threads:
Thread 1: [37, 42]
Thread 2: [46, 51]

Step 5: Final Parallel Search
Thread 1 finds 42 at its local index 1.

Step 6: Combine Final Results
We determine the global position by calculating:
Global index = (Segment index * Segment size) + Local index = (2 * 4) + 1 = 9
Therefore, 42 is at index 9 in the original array.

This example demonstrates how:

- The work is divided among multiple threads.
- Each thread performs a smaller, local binary search.
- Results are combined to narrow the search space.
- The process can be repeated on smaller segments if needed.

In a GPU implementation:

- Each thread would handle its own segment in parallel.
- Shared memory might be used for faster access to the local segments.
- Atomic operations or a reduction kernel would be used to combine results.

```
// Function to calculate the transpose of a matrix
std::vector<std::vector<int>> transposeMatrix(const std::vector<std::vector<int>>& matrix) {
  int rows = matrix.size();
  int cols = matrix[0].size();

  // Create a new matrix to store the transpose
  std::vector<std::vector<int>> transpose(cols, std::vector<int>(rows));

  // Loop to compute the transpose
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      transpose[j][i] = matrix[i][j];
    }
  }
  return transpose;
}
```

The code above is highly parallelizable. In one solution, each core can transpose the entire row to a column in the transposed matrix. Instead of transposing the entire row, each core can also tranpose matrix[i][j] individually and this approach is more efficient. 

If we do in-place transposition, we can allow only the cores handling the lower triangular part of the matrix to perform the swaps to prevent race condition.

```
void DFS(int start) {
  std::vector<bool> visited(V, false);
  DFSUtil(start, visited);
}

void DFSUtil(int v, std::vector<bool>& visited) {
  // Mark the current vertex as visited and print it
  visited[v] = true;
  std::cout << v << " ";

  // Recur for all the vertices adjacent to this vertex
  for (const int& neighbor: adjList[v]) {
    if (!visited[neighbor]) {
      DFSUtil(neighbor, visited);
    }
  }
}
```

The code above is not suitable for parallelization because DFS explores the paths sequentially. Each step depends on the result of previous steps. There is a strong sequential dependency because the decision of whether the node should be visited in the next step depends on whether its niehgbors are already visited. Also, we see recursion in the algorithm which is difficult to parallelize in GPU. Also, like in the linked-list, we don't know the number of elements in the data structure in advance. So, the amount of work per node (number of unvisited neighbors) is not known in advance and can vary greatly. This unpredictability makes it difficult to distribute work evenly across parallel processors.

```
cv::Mat colorImage = cv::imread("image.jpg");

// Create a grayscale image with the same size as the color image
cv::Mat grayImage = cv::Mat::zeros(colorImage.size(), CV_8UC1);

// Iterate through each pixel to convert to grayscale
for (int i = 0; i < colorImage.rows; ++i) {
  for (int j = 0; j < colorImage.cols; ++j) {

    // Get the pixel value (BGR format)
    cv::Vec3b color = colorImage.at<cv::Vec3b>(i, j);

    // Convert to grayscale using the formula:
    // gray = 0.299 * R + 0.587 * G + 0.114 * B
    uchar grayValue = static_cast<uchar>(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]);

    // Set the grayscale value
    grayImage.at<uchar>(i, j) = grayValue;
  }
}
```

Lastly, the code above algorithm is highly parallelizable because each pixel's conversion is independent of others and this makes it ideal for parallel processing.

## Lecture 2

Before GPUs, transformations were done on the CPU. 3D objects are represented by vertices (points in 3D space). Vertices are represented by their X, Y, and Z coordinates and they are connected to form polygons (a figure with at least three straight sides and angles). The 3D model data that is defined by vertices is converted into 2D pixels on a screen. Before GPUs, all the operations that are needed to convert the 3D coordinates to 2D screen positions were performed by the CPU. This was not efficient because CPU had to calculate color and properties of each pixel one at a time sequentially. 

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

### The Birth of GPU Computing

One of the important goals for designing chip is programmability. They should be easy to program.

- **Step 1:** Designing GPUs in such a way that they perform both floating-point and integer operations efficiently.
- **Step 2:** Increasing the number of processors to increase data parallelism.
- **Step 3:** Adding large caches, memory, and control logic to shader processors to make them fully programmable, to increase their flexibility and allow them to run more complex and varied programs (not just predefined graphics operations)
- **Step 4:** Desiginng multiple shader processors were designed to share caches and control logic to reduce hardware costs and to open up more space for the  execution unit.
- **Step 5:** Memory load/store instructions were added to allow GPUs to access memory more flexibly, similar to CPUs. This was crucial for many non-graphics algorithms.
- **Step 6:** CUDA, a software layer, is created to allow programmers to use C/C++ to write programs for GPUs. This made GPU computing accessible to a wide range of programmers.

### SISD, MISD, SIMD, MIMD

<img width="706" alt="image" src="https://github.com/user-attachments/assets/6952d19e-26c9-4987-89f3-530e7769851d">

In the image above, we see 4 different computer architecture models. 

1) SISD (Single Instruction, Single Data): In this design, one processing unit (processor/core) operates on a single data stream. (Stream in here means a sequence of elements that are processed one after another. These elements could be numbers, pixels, characters, etc. Data stream refers to the input that the processing unit is working on. For instance, if we have image and we want to do image processing, a data stream might be the sequence of pixel values in an image). This design represents a traiditonal, sequential computer architecture.

2) SIMD (Single Instruction, Multiple Data): In this design, one processing unit operates on different parts of the data simultaneously. This design is common in modern GPUs since it is basically parallel processing of data with the same instruction.

3) MISD (Multiple Instruction, Single Data): Multiple instruction streams operate on a single data stream. THis is rarely used. (Instruction stream is a sequence of operations that the processor need to execute. These are the instructions that tell the processing unit what to do with the data. For instance, an instruction stream might contain operations like "add", "multiply", "load from memory", etc.

4) MIMD (Multiple Instruction, Multiple Data): Multiple instruction streams operate on multiple data streams. This is the most flexible and powerful design. It repreesnts modern multicore processors and parallel computing systems. It can also be seen as fully parallel processing with different instructions on different data.

### Problems Faced by GPU 

1) Under Utilization: Not all tasks are suitable for parallelism. These tasks leave many of the processing units in GPU idle.  
2) Bandwidth to CPU: GPUs often need to receive data from the CPU to process it and return the processed data back to the CPU. The speed of this data transfer can limit overall performance.
3) Memory Access: Memory has been a problem in computing from the very beginning until now since it is slow. The memory is about 1000 times slower than your processor and the disk is 1000 times slower than the memory.

### Hardware 

<img width="687" alt="image" src="https://github.com/user-attachments/assets/97a7113c-2fd6-498d-9923-bffedf6a89b6">

North bridge is a chip that acts as a communication hub. It manages data transfers between the CPU, RAM, and other high-performance components such as GPU. And it does this by using the memory controller that is located inside of it.

Memory controller, on the other hand, is a component that is integrated to the CPU or GPU. It contains lots of transistors and coordinates reading from/writing to RAM. 

<img width="300" alt="image" src="https://github.com/user-attachments/assets/d4bbcc88-9002-474e-a5f3-15d4000dfc25"> \\

In the two pictures below, we can see the interaction between northbridge, memory controller, and many other components of the computer in more detail. 

<img width="300" alt="image" src="https://github.com/user-attachments/assets/4e5d2e2a-0bf8-45ed-9f08-842f153c10ca">

<img width="300" alt="image" src="https://github.com/user-attachments/assets/bdb7140a-9253-454b-acb3-e0b7fe8a0f5e">

In addition to these, in the picture below, we see a multi CPU computer architecture with an integrated IO hub. Each CPU has its own directly connected memory. This allows faster access to local memory. We also see a direct link between the two CPUs. This allows for direct communication and data sharing between them. This is important for maintaining cache coherency and efficient parallel processing. 

The IO hub is a component that manages various input and output operations. It acts like a traffic controller for data that moves between different parts of the system. The IO hub is connected to the GPU via a PCI Express. 

<img width="531" alt="image" src="https://github.com/user-attachments/assets/2fe6f9d2-7bfa-40f8-8d3b-512804925c0f">

In the picture below, we see GPU, the architecture of graphics card, and its connection to the main system components. 

<img width="443" alt="image" src="https://github.com/user-attachments/assets/195c8f28-fd44-4838-bbc5-e68d5e753218">

- Host CPU is the main processor of the system.
- As we mentioned before, north bridge is a chip that manages communications between the CPU, system memory, and other high-speed components using the memory controller that is located inside of it. In newer systems, these functions are typically integrated into the CPU.
- System DRAM is the main system memory (RAM) used by the CPU.
- GDDR (Graphics Double Data Rate memory) is a memory chip that is optimized for rapid data transfer between the GPU and GPU memory. It serves as the dedicated memory for the GPU and it is separate from the system's main memory. Through this separation, GPU can process graphics data without competing with the CPU for memory access. GDDR stores textures, frame buffers, and other graphics-related data that the GPU needs to access quickly to during rendering.

### PCIe 

PCIe is a component that provides high-speed communication between the motherboard and various peripherals such as SSDs, network cards, etc. The data transfer that is made through PCIe is coordinated by the clock signal. This ensures that the sender and receiver operate in sync. 

In PCIe, data bits are sent one after another over a single channel. This is known as serial communication. Through serial commnunication, fewer wires are needed, it becomes easier to maintain signal integrity at high speeds on a single channel, and it becomes easier to increase bandwidth by adding more lanes or inceasing the signaling rate. Also, with parallel communication, the bits may arrive due to difference in wire length, eletromagnetic interference or signal degradation. This issuse is not the case for serial communication. Because only one bit is sent at a time, PCIe can run at much higher clock frequencies and achieve higher data rates.

These factors make PCIe highly efficient and suitable for modern high-speed data transfer needs.

Also, in PCIe, each connection is directly between two devices and this connection is not shared among multiple devices. This is also named as pont-to-point connection. Through this way, it becomes easier to add more devices without affecting the performance of the existing devices. Also, the total amount of data that can be carried through PCIe (bandwidth) is limited. Because each connection is directly between two devices and no connection is shared among multiple devices, each device gets its own dedicated connection the PCIe and this ensures that devices don't have to compete for bandwidth. 

<img width="454" alt="image" src="https://github.com/user-attachments/assets/5254c8c2-af53-4292-aa4f-dfb909c09306">

<img width="508" alt="image" src="https://github.com/user-attachments/assets/72eb6022-4b82-48ee-940c-2d5c84ed0bae">

<img width="749" alt="image" src="https://github.com/user-attachments/assets/e7aca415-6efa-4070-8bae-ef0c3e9a0574">

Also, PCIe switch is a device that expands the connectivity options of PCIe interfaces.

<img width="1101" alt="image" src="https://github.com/user-attachments/assets/21e31c5b-ddee-4c0d-9b94-cdc7e93609c9">

It contains multiple sets of PCIe lanes. It can take a single upstream port (connected to the host system) and divide it into multiple downstream ports. The switch examines the headers of PCIe packets to determine their destination. It then routes these packets to the appropriate downstream port or back to the upstream port.

Switches include buffers to temporarily store data packets. This allows them to manage traffic flow when different devices operate at different speeds or when there's congestion.

When multiple devices try to communicate simultaneously, the switch arbitrates access to the upstream port, ensuring fair and efficient use of bandwidth.

The switch manages PCIe protocol details like flow control, error handling, and power management for connected devices.

It can perform address translation between the system's memory space and the address spaces of connected devices, enabling efficient communication.

Advanced switches can prioritize traffic from certain ports or devices, ensuring critical data gets through first.

These mechanisms allow the switch to effectively manage multiple PCIe connections, expanding connectivity while maintaining performance. 

### NVLink 

NVLink is a high-bandwidth interconnect technology developed by NVIDIA. It's designed to enable fast communication between GPUs (Graphics Processing Units) and, in some cases, between GPUs and CPUs. 

<img width="663" alt="image" src="https://github.com/user-attachments/assets/062a7b21-e106-4e7a-a400-9e8dbfe35f89">

NVLink aims to overcome the bandwidth limitations of traditional PCIe (Peripheral Component Interconnect Express) connections, allowing for much faster data transfer between GPUs.

It can provide significantly higher bandwidth compared to PCIe, with exact speeds varying by generation. For example, NVLink 3.0 can offer up to 600 GB/s bidirectional bandwidth between GPUs.

It's particularly useful in high-performance computing, deep learning, and AI applications where multiple GPUs need to work together efficiently.

NVLink allows for more flexible and efficient scaling of multi-GPU systems, improving overall system performance in GPU-intensive tasks.

In some implementations, NVLink can also connect GPUs directly to CPUs, further enhancing system performance.

NVLink plays a crucial role in enabling more powerful and efficient multi-GPU computing systems, particularly for demanding applications in scientific research, AI, and data analytics.

### Modern GPU Hardware

<img width="533" alt="image" src="https://github.com/user-attachments/assets/f512dce5-91fb-4d8e-afed-0b1b57240c20">

Why there is L2 cache while there is a shared memory and global memory ? What are the things that make L2 cache necessary while shared memory and global memory exist ?

Global memory, while large, is relatively slow to access. Shared memory is fast but small and limited to each SM. L2 cache acts as an intermediate layer, bridging the speed gap between the fast but limited shared memory and the slow but large global memory.

Frequent accesses to global memory can create a bottleneck. L2 cache helps reduce this pressure by storing frequently accessed data closer to the SMs. This reduces the number of times the GPU needs to access the high-latency global memory.

Shared memory is local to each SM and can't be directly accessed by other SMs. L2 cache provides a way for different SMs to share data more efficiently than going through global memory every time.

Shared memory is limited in size and managed by the programmer or compiler. L2 cache automatically stores recently used data that may not fit or may not be explicitly placed in shared memory.

While shared memory is great for predictable access patterns, L2 cache can improve performance for more irregular memory access patterns that are common in many algorithms.

L2 cache helps hide the latency of global memory accesses by providing faster access to recently used data.

In summary, while shared memory provides fast, explicitly managed storage for each SM, and global memory provides large capacity, the L2 cache plays a crucial role in balancing performance, efficiency, and ease of use. It helps manage data that doesn't fit neatly into the shared memory model while significantly reducing the performance penalty of accessing global memory. This three-tiered memory structure (shared memory, L2 cache, global memory) allows GPUs to maintain high performance across a wide range of applications and access patterns.

<img width="652" alt="image" src="https://github.com/user-attachments/assets/3ce33f40-09b2-4134-9812-7feeb75af4fe">

### Scalar Program vs Vector Program

<img width="266" alt="image" src="https://github.com/user-attachments/assets/8d6d565e-27ff-4b39-9808-e7682a6b70cf">
<img width="704" alt="image" src="https://github.com/user-attachments/assets/a861f804-925c-4307-ac87-f69da050e02f">

These images illustrate the difference between scalar and vector programming approaches, particularly in the context of parallel processing.

Image 1 shows a scalar program:
1. It declares a 2D array A[4][8] of floats.
2. It uses nested for loops to iterate over each element of the array.
3. The outer loop (i) iterates 4 times, and the inner loop (j) iterates 8 times.
4. Each iteration increments the value of A[i][j].

This is a traditional scalar approach where operations are performed sequentially on individual data elements.

Image 2 explains vector programming:
1. "Vector width is exposed to programmers" means that programmers can explicitly work with groups of data elements (vectors) rather than individual elements.
2. It compares scalar vs. vector approaches:
   - Scalar: processes one element (A[i][j]) at a time using a single Processing Element (PE).
   - Vector: processes multiple elements simultaneously using multiple PEs.
3. The image shows a "Vector of width 8", where 8 elements (A[i][0] to A[i][7]) are processed in parallel by 8 PEs (PE 0 to PE 7).
4. "Vectorizing compilers are doing good job, if helped by the programmer!" suggests that modern compilers can automatically vectorize code, but programmer assistance can improve this process.

Key points:
1. Scalar programming processes one data element at a time.
2. Vector programming processes multiple data elements simultaneously, utilizing parallel processing capabilities of modern hardware.
3. Vector programming can significantly improve performance for operations on large datasets, especially in fields like scientific computing, image processing, and machine learning.
4. While compilers can automatically vectorize some code, programmer awareness and optimization can lead to better vectorization and performance.

### Kernel 

A kernel is a function that will be executed by a thread on the GPU. Kernels define the computations to be performed by each thread on the GPU and they typically perform a small amount of work per thread. When a kernel is launched, it's executed simultaneously by many threads on the GPU. Kernels are defined within the application code and the application that is running on the CPU launches kernels to execute on the GPU. They allow programmers to express parallelism by specifying the operations that each thread should perform. They are designed to take advantage of the GPU's massive parallelism. 

Here are some examples of **GPU kernels** in CUDA:

#### 1. **Vector Addition Kernel**
This is a simple example where two arrays (vectors) are added element-wise.

```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```
- **What it does**: Adds two arrays `A` and `B` element by element and stores the result in array `C`.
- **How it works**: Each thread computes one element of the result. The total number of threads matches the size of the vectors.

#### 2. **Matrix Multiplication Kernel**
This example performs matrix multiplication for two 2D matrices.

```cpp
__global__ void matrixMul(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
```
- **What it does**: Multiplies two matrices `A` and `B` and stores the result in matrix `C`.
- **How it works**: Each thread is responsible for computing one element of the output matrix. Threads in a block work together to compute different rows and columns of the result matrix.

#### 3. **Reduction Kernel (Sum of Array)**
This kernel sums all the elements in an array using a parallel reduction.

```cpp
__global__ void reduceSum(float *input, float *output, int N) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory
    sharedData[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // Write the result of this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}
```
- **What it does**: Computes the sum of all elements in an array.
- **How it works**: Each thread block computes the partial sum of a portion of the array, storing the result in shared memory. Then, these partial sums are combined to get the final result.

#### 4. **Image Blur Kernel (2D Convolution)**
This kernel performs a simple image blurring operation using a 2D convolution.

```cpp
__global__ void blurImage(const unsigned char *inputImage, unsigned char *outputImage, int width, int height, const float *filter, int filterWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float blurValue = 0.0f;
        for (int filterY = 0; filterY < filterWidth; ++filterY) {
            for (int filterX = 0; filterX < filterWidth; ++filterX) {
                int imageX = min(max(x + filterX - filterWidth / 2, 0), width - 1);
                int imageY = min(max(y + filterY - filterWidth / 2, 0), height - 1);
                blurValue += inputImage[imageY * width + imageX] * filter[filterY * filterWidth + filterX];
            }
        }
        outputImage[y * width + x] = blurValue;
    }
}
```
- **What it does**: Applies a blurring filter to an image by performing 2D convolution.
- **How it works**: Each thread processes one pixel by applying a convolution filter over a window of surrounding pixels.

#### 5. **Monte Carlo Simulation Kernel**
A Monte Carlo simulation kernel that estimates the value of Pi by randomly sampling points inside a unit square.

```cpp
__global__ void estimatePi(int *count, int numSamples, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localCount = 0;
    curandState state;
    curand_init(seed, tid, 0, &state);

    for (int i = tid; i < numSamples; i += blockDim.x * gridDim.x) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f) {
            localCount++;
        }
    }

    atomicAdd(count, localCount);
}
```
- **What it does**: Estimates the value of Pi by counting random points that fall inside a unit circle.
- **How it works**: Each thread generates random points and checks if they lie within a unit circle, then accumulates the result.

---

These are just a few examples of CUDA kernels that perform different computations. In each case, the kernel is a function executed in parallel by many threads on the GPU.

### Multithreaded in GPUs 

<img width="665" alt="image" src="https://github.com/user-attachments/assets/8295781d-539f-4bff-95bc-31119dfc2368">

This image illustrates the concept of multithreading in GPUs, specifically focusing on the organization of threads and blocks in a GPU's parallel processing structure. Let's break it down in more detail:

The image shows a hierarchy: Grid > Blocks > Threads. It demonstrates a configuration of (4x1) blocks, each containing (8x1) threads

The top level is called a "Grid". In this example, the Grid contains a single row of 4 blocks (4x1). The Grid represents the entire computation space for a GPU kernel. 

There are 4 blocks in the Grid, labeled as "block 0,0" to "block 0,3". Blocks are units of work that can be distributed across the GPU's streaming multiprocessors. Each block can be processed independently, allowing for scalability across different GPU architectures

The image zooms in on one block to show its internal structure. Each block contains 8 threads in a single row (8x1 configuration). Threads within a block can cooperate and share resources

Threads are the smallest units of parallel execution. In this example, there are 8 threads per block, numbered from 0,0 to 0,7. Each thread executes the same kernel code but typically operates on different data

This structure allows for two levels of parallelism:
  a) Between blocks: Different blocks can execute on different streaming multiprocessors
  b) Within blocks: The 8 threads in each block can execute in parallel on a single multiprocessor

This organization allows GPU programs to scale across different GPU architectures. More powerful GPUs can process more blocks simultaneously

This structure is typical of NVIDIA's CUDA programming model, where programmers define kernel functions that are executed across this grid of blocks and threads

Understanding this structure is crucial for efficient GPU programming, as it allows developers to organize their computations to take full advantage of the GPU's parallel processing capabilities. The programmer needs to divide the problem into appropriate grid, block, and thread configurations to achieve optimal performance.

### Scheduling Thread Blocks on SM 

<img width="725" alt="image" src="https://github.com/user-attachments/assets/90eacc3e-a451-478f-8c8c-e29f556e0c13">

The picture above shows how the 4 blocks from the grid are distributed across the 3 available SMs:
- The thread block (1,1) and (0,0) are assigned to SM 1 to be executed.
- The thread block (0,1) is assigned to SM 2 to be executed.
- The thread block (1,0) is assigned to SM 3 to be executed.

This layout illustrates several important concepts in GPU computing:

1. Parallelism: Multiple blocks can run concurrently on different SMs.
2. Scalability: The same kernel can run on GPUs with different numbers of SMs.
3. Hardware abstraction: Programmers work with a logical grid of threads, while the GPU handles the physical distribution of work.
4. Efficient resource utilization: The GPU can balance work across available SMs, even when the number of blocks doesn't evenly divide the number of SMs.

### GPU Programming Model 

- Application can include multiple kernels.
- Threads of the same block run on the same SM.
  - Block in an SM is divided into warps of 32 threads each.
  - A warp is the fundamental unit of dispatch in an SM (dispatch unit refers to the smallest group of threads that the GPU schedules and executes together as a single unit. In NVIDIA GPUs, this unit is the warp).
- Blocks in a grid can coordinate using global memory
- Each grid executes a kernel. 

### Scheduling In Modern NVIDIA GPUs

GPUs typically work on tasks from one application at a time. There is not an exclusive access to GPU resources for a single application. Context switch time when the GPU changes from executing tasks for one application to another is ~25 microseconds. This quick switch time allows for efficient sharing of GPU resources among multiple applications, even if not truly simultaneous. 

Modern GPUs support concurrent kernel execution. Different kernels from the same application can run at the same time on different parts of the GPU. This feature allows for better utilization of GPU resources and can improve overall application performance. 

Warps (groups of 32 threads) from different blocks or even different kernels can be actively executing at the same time. This capability allows for high levels of parallelism and efficient use of GPU resources.

1. Parallelism: GPUs can handle multiple levels of parallelism - within a kernel, across kernels, and even across applications (with some limitations).
2. Resource Utilization: The ability to run multiple kernels and warps from different blocks simultaneously allows for better utilization of GPU resources.
3. Flexibility: The GPU can adapt to different workloads, whether they involve a single complex kernel or multiple smaller kernels.
4. Application Design: Developers can design applications to take advantage of concurrent kernel execution, potentially improving overall performance.
5. Scheduling Complexity: The GPU's scheduler must manage this complex execution environment, balancing resources across multiple kernels and blocks.
6. Time-Sharing: While primarily dedicated to one application, the quick context switch time allows for effective time-sharing of the GPU among multiple applications.

These features contribute to the high performance and flexibility of modern GPUs, allowing them to efficiently handle a wide range of computational tasks across various applications.

There are two types of scheduling process: scheduling thread blocks to various SMs, and distribution the warps of 32 threads to the execution units. 

#### Multi-Process Service (MPS) 

GPUs (Graphics Processing Units) contain many hardware resources that programs can use for computation. However, it's often difficult or impossible to develop programs that fully utilize all these resources, leading to what's known as the "under-utilization" problem. One potential solution to this problem is to run multiple applications on the same GPU simultaneously, which could increase overall resource utilization. However, there are several challenges to implementing this solution:

- Unlike CPUs, GPUs lack fine-grained sharing mechanisms. This means they don't have sophisticated ways to divide their resources among multiple tasks.
- GPUs don't have virtual memory, a feature in CPUs that allows multiple programs to share memory safely and efficiently.
- Context switching (changing from one task to another) is very expensive in terms of performance on GPUs. This is because GPUs typically work with large amounts of data, and moving this data around during a switch takes a lot of time and resources.

To address these issues, NVIDIA introduced Hyper-Q technology. This technology allows multiple CPU threads to launch work on a single GPU, effectively improving the GPU's ability to handle multiple tasks.

<img width="737" alt="image" src="https://github.com/user-attachments/assets/cfaec691-d655-48eb-8d2b-8737c8221e9f">

### Step-by-Step Process of Executing a Kernel on a GPU

Let's walk through the process of executing a kernel on a GPU, which involves threads, blocks, and warps. We'll use a simple example of adding two large vectors element-wise.

**Step 1: Kernel Definition**\
First, we define a kernel function that will be executed on the GPU. This function describes the operation each thread will perform.

```cuda
__global__ void vectorAdd(float* A, float* B, float* C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}
```

**Step 2: Kernel Launch**\
The host (CPU) launches the kernel, specifying the number of blocks and threads per block.

```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
```

**Step 3: GPU Scheduling**\
The GPU has a scheduler that decides which blocks go to which SMs. The GPU's scheduler receives the kernel launch command and begins distributing blocks to available Streaming Multiprocessors (SMs). Blocks are groups of threads that can communicate with each other, that can share data using shared memory, and that can be easily synchronized with each other. GPUs have multiple SMs and each of these SMs is capable of running blocks independently. By distributing blocks of SMs, we can make a better use of the GPU's total computational sources. Blocks are typically assigned to SMs as they become available, not all at once. The scheduler tries to keep all SMs busy by distributing blocks evenly. The scheduler takes into account each SM's available resources (registers, shared memory) when assigning blocks.

**Step 4: Block Assignment**\
Each SM is assigned one or more blocks. The SM is responsible for executing all threads within its assigned blocks.

**Step 5: Warp Formation**\
Within each block, threads are grouped into warps (typically 32 threads per warp). These warps are the fundamental unit of execution on the GPU.

**Step 6: Warp Scheduling**\
The SM's warp scheduler selects warps that are ready to execute and issues instructions to them.

**Step 7: Instruction Execution**\
All threads in a warp execute the same instruction simultaneously (SIMT - Single Instruction, Multiple Thread).

- In our example, each thread calculates its global index `i`.
- It checks if `i` is within the vector's size.
- If so, it performs the addition `C[i] = A[i] + B[i]`.

**Step 8: Memory Access**\
As threads perform memory operations, the GPU's memory system handles these requests:
- Global memory accesses are coalesced when possible for efficiency.
- Shared memory might be used for data that's accessed multiple times within a block.

**Step 9: Warp Divergence Handling**\
If threads within a warp take different paths (e.g., some threads have `i < size` while others don't), the warp executes both paths, disabling threads that don't need to execute each path.

**Step 10: Warp Completion**\
As warps complete their execution, the SM's resources become available for other warps.

**Step 11: Block Completion**\
When all warps in a block have completed, the block's resources are freed, and the SM can begin executing another block if available.

**Step 12: Kernel Completion**\
The kernel execution is complete when all blocks have finished processing.

**Step 13: Result Availability**\
The results of the computation (in our case, the sum vector C) are now available in GPU memory and can be copied back to the host if needed.

This process demonstrates how the concepts of threads, blocks, warps, and kernels work together to execute parallel computations on a GPU. The hierarchical organization allows for efficient scheduling and execution across different levels of parallelism, from individual threads up to the entire grid of blocks.

### Fermi 

<img width="748" alt="image" src="https://github.com/user-attachments/assets/318a878b-9422-44ae-9d0e-e89655bf7f87">

Let's say we're performing the operation: A + B = C, where A and B are floating-point numbers.

1. Dispatch Port:
   - The dispatch port receives the instruction from the warp scheduler.
   - Example: It receives the instruction to add two floating-point numbers.

2. Operand Collector:
   - This unit gathers the necessary data (operands) for the operation.
   - It fetches the values of A and B from registers or memory.
   - Example: It collects the values A = 3.14 and B = 2.86.

3. FP Unit (Floating Point Unit):
   - Since our operation involves floating-point numbers, the FP unit will be used.
   - It performs the actual computation.
   - Example: The FP unit adds 3.14 and 2.86, resulting in 6.00.

4. INT Unit (Integer Unit):
   - While not used in our floating-point example, this unit would handle integer operations.
   - It would be used for operations like integer addition, bitwise operations, etc.

5. Result Queue:
   - After the computation is complete, the result is placed in the result queue.
   - From here, it can be written back to registers or memory.
   - Example: The result 6.00 is placed in the queue to be written to the destination (C).

Workflow example:

1. The warp scheduler issues an add instruction to the Dispatch Port.
2. The Dispatch Port sends this instruction to the Operand Collector.
3. The Operand Collector fetches the values 3.14 (A) and 2.86 (B) from registers.
4. These operands are sent to the FP Unit along with the add instruction.
5. The FP Unit performs the addition: 3.14 + 2.86 = 6.00.
6. The result (6.00) is placed in the Result Queue.
7. From the Result Queue, the result is written back to the register or memory location for C.

This process happens in parallel across multiple CUDA cores, allowing for high-throughput computation. The INT Unit would be used similarly for integer operations, following the same general flow but performing integer arithmetic instead of floating-point arithmetic.

The entire process is designed for efficiency, allowing the GPU to perform massive numbers of similar operations in parallel, which is key to its performance in tasks like graphics rendering, scientific simulations, and machine learning computations.

### Memory Hierarchy 

The GPU uses a 40-bit address space, which allows for 2^40 = 1,099,511,627,776 bytes (1 terabyte) of addressable memory. This large, unified address space allows for flexible memory management across different types of GPU memory.

A unified address space in the context of GPU memory hierarchy refers to a more consistent and integrated way of handling different types of memory. All types of memory (global, shared, local) are part of the same 40-bit address space and this means there's a single, coherent way to reference memory locations, regardless of the memory type. Therefore, the same load/store instructions can be used to access different types of memory and programmers don't need to use separate instruction sets for different memory types, simplifying the programming model. Also, in unified address space, it's easier to move data between different memory types or to change how memory is used in a program without major code restructuring. Developers can think of memory more holistically, rather than as completely separate systems. This can reduce the complexity of memory management in GPU programming.

Imagine you're developing a program that initially uses global memory for a certain data structure. With a unified approach, if you later decide that this data would benefit from being in shared memory for performance reasons, the change might be as simple as altering the memory allocation, without needing to change how the data is accessed in your code.

While this unified approach doesn't eliminate all the complexities of GPU memory management (developers still need to be aware of performance implications of different memory types), it does provide a more streamlined and flexible framework for working with GPU memory.

As we mentioned, there are three types of memory within this address space:

- Global Memory:
  - Accessible by all threads across all streaming multiprocessors (SMs).
  - Typically the largest but slowest memory.

- Shared Memory:
  - Shared among threads within the same thread block.
  - Faster than global memory, but smaller in size.

- Local Memory:
  - Private to each thread.
  - Used for thread-specific variables that don't fit in registers.

All these memory types can be accessed using the same set of load and store instructions. This uniformity simplifies programming and allows for more flexible memory usage.

## Lecture 3 
- There is no branch prediction for GPU to leave space for more execution unit.
- Kernel is a peice fo code that goes to GPU
- Every thread execute a kernel
- Every thread has a unique ID in block
- Blocks also has to have some ID (we decide these IDs)
- The group of blocks is called grid
- All blocks must be of the same size
- All threads in all the blocks should execute the same kernel
- Assume that each SM has 32 SPs. Once is block assigned to SM and saved in there until execution finishes. One block goes to SM. That lbock has 32 SPs but the block has
- I didn't understand why there are two groupings of threads: warp and block.
- L1 is programmer transparent. That's why there is a shared memory in an SM and the SPs can communicate over shared memory instead of L1. You can control shared memory by hand. Global memory is also controllable by the user. You decide what to put inside the shared memory.
- Kernels from the same application reads from the same global memory.
- MPS is alternative to CUDA APIs. Designed to enable multi-process CUDA applications to utilizie Hyper-Q capatibilites (Hyper-Q is a feature in NVIDIA's Kepler and later GPUs that allows multiple CPU cores to simultaneously send work to a single GPU). Allows sevral different applications to use the GPU at any point
- Two level of scheduler: thread scheduler and warp scheduler.
- Registers are quite different in CPU and GPU. Every SM has thousands of registers in GPU. Registers are under user controller indirectly.
- Cache coherency is not supported in GPU. If two applications access the same information simultaneously in the global memory, that would cause chaos.
- All addresses in the GPU are allocated from a continuous 40-bit address space.
- Global, shared, and local addresses are defined as ranges within this address space and can be accessed by common load/store instructions. These load/store instructions support 64 bit addresses to allow for future growth.
- CUDA is on top of C, C++, Fortran.
- GPU parllelism (number of SPs or SMs) is doubling almost every year.


