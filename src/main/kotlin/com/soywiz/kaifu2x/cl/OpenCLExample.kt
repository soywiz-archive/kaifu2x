package com.soywiz.kaifu2x.cl

import org.jocl.*
import org.jocl.CL.*


object OpenCLExample {
    private val programSource = """
        __kernel void sampleKernel(__global const float *a, __global const float *b, __global float *c) {
            int gid = get_global_id(0);
            c[gid] = a[gid] * b[gid] * b[gid] * b[gid] * b[gid] * b[gid] * b[gid] * b[gid] * b[gid] * b[gid];
        }
    """

    @JvmStatic
    fun main(args: Array<String>) {

        // Create input- and output data
        val n = 10000
        val srcArrayA = FloatArray(n)
        val srcArrayB = FloatArray(n)
        val dstArray = FloatArray(n)
        for (i in 0 until n) {
            srcArrayA[i] = i.toFloat()
            srcArrayB[i] = i.toFloat()
        }
        val srcA = Pointer.to(srcArrayA)
        val srcB = Pointer.to(srcArrayB)
        val dst = Pointer.to(dstArray)

        // The platform, device type and device number
        // that will be used
        val platformIndex = 0
        val deviceType = CL_DEVICE_TYPE_ALL
        val deviceIndex = 0

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true)

        // Obtain the number of platforms
        val numPlatformsArray = IntArray(1)
        clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]

        // Obtain a platform ID
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        clGetPlatformIDs(platforms.size, platforms, null)
        val platform = platforms[platformIndex]

        // Initialize the context properties
        val contextProperties = cl_context_properties()
        contextProperties.addProperty(CL_CONTEXT_PLATFORM.toLong(), platform)

        // Obtain the number of devices for the platform
        val numDevicesArray = IntArray(1)
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]

        // Obtain a device ID
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
        val device = devices[deviceIndex]!!

        // Create a context for the selected device
        val context = clCreateContext(
                contextProperties, 1, arrayOf<cl_device_id>(device), null, null, null)

        // Create a command-queue for the selected device
        val commandQueue = clCreateCommandQueue(context, device, 0, null)

        // Allocate the memory objects for the input- and output data
        val memObjects = arrayOfNulls<cl_mem>(3)
        memObjects[0] = clCreateBuffer(context,
                CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR,
                (Sizeof.cl_float * n).toLong(), srcA, null)
        memObjects[1] = clCreateBuffer(context,
                CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR,
                (Sizeof.cl_float * n).toLong(), srcB, null)
        memObjects[2] = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                (Sizeof.cl_float * n).toLong(), null, null)

        // Create the program from the source code
        val program = clCreateProgramWithSource(context,
                1, arrayOf(programSource), null, null)

        // Build the program
        clBuildProgram(program, 0, null, null, null, null)

        // Create the kernel
        val kernel = clCreateKernel(program, "sampleKernel", null)

        // Set the arguments for the kernel
        clSetKernelArg(kernel, 0,
                Sizeof.cl_mem.toLong(), Pointer.to(memObjects[0]))
        clSetKernelArg(kernel, 1,
                Sizeof.cl_mem.toLong(), Pointer.to(memObjects[1]))
        clSetKernelArg(kernel, 2,
                Sizeof.cl_mem.toLong(), Pointer.to(memObjects[2]))

        // Set the work-item dimensions
        val global_work_size = longArrayOf(n.toLong())
        val local_work_size = longArrayOf(1)

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_size, local_work_size, 0, null, null)

        // Read the output data
        clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                (n * Sizeof.cl_float).toLong(), dst, 0, null, null)

        // Release kernel, program, and memory objects
        clReleaseMemObject(memObjects[0])
        clReleaseMemObject(memObjects[1])
        clReleaseMemObject(memObjects[2])
        clReleaseKernel(kernel)
        clReleaseProgram(program)
        clReleaseCommandQueue(commandQueue)
        clReleaseContext(context)

        // Verify the result
        var passed = true
        val epsilon = 1e-7f
        for (i in 0 until n) {
            val x = dstArray[i]
            val y = srcArrayA[i] * srcArrayB[i]
            val epsilonEqual = Math.abs(x - y) <= epsilon * Math.abs(x)
            if (!epsilonEqual) {
                passed = false
                break
            }
        }
        println("Test " + if (passed) "PASSED" else "FAILED")
        if (n <= 10) {
            println("Result: " + java.util.Arrays.toString(dstArray))
        }
    }
}