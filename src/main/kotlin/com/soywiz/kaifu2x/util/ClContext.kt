package com.soywiz.kaifu2x.util

import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.lang.UTF8
import com.soywiz.korio.lang.toString
import org.intellij.lang.annotations.Language
import org.jocl.*
import org.jocl.CL.*
import java.io.Closeable
import java.nio.*

class ClContext(val type: DeviceType = DeviceType.PREFER_GPU) : Closeable {
    enum class DeviceType {
        ANY, PREFER_GPU, FORCE_GPU
    }

    internal val context: cl_context
    internal val platform: cl_platform_id
    internal val device: cl_device_id

    init {
        setExceptionsEnabled(true)

        platform = getPlatformIds().first()
        device = when (type) {
            DeviceType.ANY -> platform.getDevices(CL_DEVICE_TYPE_ALL).first()
            DeviceType.PREFER_GPU -> platform.getDevices(CL_DEVICE_TYPE_GPU).firstOrNull() ?: platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR).firstOrNull() ?: platform.getDevices(CL_DEVICE_TYPE_ALL).first()
            DeviceType.FORCE_GPU -> platform.getDevices(CL_DEVICE_TYPE_GPU).firstOrNull() ?: invalidOp("Can't find GPU devices")
        }
        println("OpenCL Device: " + device)

        // Create a context for the selected device
        context = clCreateContext(
                cl_context_properties().apply { addProperty(CL_CONTEXT_PLATFORM.toLong(), platform) },
                1, arrayOf(device), null, null, null
        )
    }

    private val cl_device_id.name: String get() = getDeviceInfoString(this, CL_DEVICE_NAME)

    private fun cl_platform_id.getDevices(deviceType: Long): List<cl_device_id> {
        val numDevicesArray = IntArray(1)
        clGetDeviceIDs(this, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        clGetDeviceIDs(this, deviceType, numDevices, devices, null)
        return devices.filterNotNull()
    }

    private fun getPlatformIds(): List<cl_platform_id> {
        val numPlatformsArray = IntArray(1)
        clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        //println("nplatforms: ${platforms.size}")
        clGetPlatformIDs(platforms.size, platforms, null)
        return platforms.filterNotNull()

    }

    private fun getDeviceInfoString(device: cl_device_id, param_name: Int): String {
        val out = ByteArray(1024)
        val outSize = LongArray(1)
        clGetDeviceInfo(device, param_name, out.size.toLong(), Pointer.to(out), outSize)
        return out.copyOf(outSize[0].toInt()).toString(UTF8)
    }

    fun createCommandQueue() = ClCommandQueue(this)
    fun createBuffer(data: FloatArray, size: Int = data.size, writeable: Boolean = false) = ClBuffer(this, Pointer.to(data), Sizeof.cl_float, size, writeable)
    fun createBuffer(data: FloatBuffer, size: Int = data.limit(), writeable: Boolean = false) = ClBuffer(this, Pointer.to(data), Sizeof.cl_float, size, writeable)
    fun createBuffer(data: IntBuffer, size: Int = data.limit(), writeable: Boolean = false) = ClBuffer(this, Pointer.to(data), Sizeof.cl_int4, size, writeable)
    fun createEmptyBuffer(elementSize: Int, length: Int, writeable: Boolean = true) = ClBuffer(this, null, elementSize, length, writeable = true)
    fun createProgram(@Language("opencl") source: String) = ClProgram(this, source)

    inline fun <T> queue(callback: ClCommandQueue.() -> T): T {
        return createCommandQueue().use { queue ->
            callback(queue)
        }
    }

    override fun close() {
        clReleaseContext(context)
    }
}

class ClBuffer(val ctx: ClContext, val ptr: Pointer?, val elementSize: Int, val length: Int, val writeable: Boolean) : Closeable {
    val sizeInBytes: Int = elementSize * length
    private val flags = run {
        var flags = 0L
        flags = flags or (if (writeable) CL_MEM_READ_WRITE else CL_MEM_READ_ONLY)
        flags = flags or (if (ptr != null) CL_MEM_COPY_HOST_PTR else 0L)
        flags
    }
    val mem = clCreateBuffer(ctx.context, flags, sizeInBytes.toLong(), ptr, null)

    //fun readIntsQueue(queue: ClCommandQueue): IntArray = queue.readByteBuffer(this)
    //fun readFloatsQueue(queue: ClCommandQueue): FloatArray = queue.readByteBuffer(this)

    fun readInts(queue: ClCommandQueue): IntBuffer = queue.readByteBuffer(this).apply { queue.waitCompleted() }.asIntBuffer()
    fun readFloats(queue: ClCommandQueue): FloatBuffer = queue.readByteBuffer(this).apply { queue.waitCompleted() }.asFloatBuffer()

    override fun close() {
        clReleaseMemObject(mem)
    }
}

class ClProgram(val ctx: ClContext, val source: String) : Closeable {
    val result = IntArray(1)
    val program = clCreateProgramWithSource(ctx.context, 1, arrayOf(source), longArrayOf(source.length.toLong()), result)

    init {
        clBuildProgram(program, 0, null, null, null, null)
    }

    operator fun get(name: String) = getKernel(name)
    fun getKernel(name: String) = ClKernel(this, name)

    override fun close() {
        clReleaseProgram(program)
    }
}

class ClKernel(val program: ClProgram, val name: String) {
    operator fun invoke(queue: ClCommandQueue, vararg args: Any, globalWorkRanges: List<LongRange>? = null, localSizes: List<Long>? = null) {
        queue(queue, *args, globalWorkRanges = globalWorkRanges, localSizes = localSizes)
        queue.waitCompleted()
    }

    fun queue(queue: ClCommandQueue, vararg args: Any, globalWorkRanges: List<LongRange>? = null, localSizes: List<Long>? = null) {
        val errorCode = IntArray(1)
        val kernel = clCreateKernel(program.program, name, errorCode)
        for ((index, arg) in args.withIndex()) {
            when (arg) {
                is Int -> clSetKernelArg(kernel, index, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(arg)))
                is Float -> clSetKernelArg(kernel, index, Sizeof.cl_float.toLong(), Pointer.to(floatArrayOf(arg)))
                is ClBuffer -> clSetKernelArg(kernel, index, Sizeof.cl_mem.toLong(), Pointer.to(arg.mem))
                else -> TODO("Unsupported argument $arg")
            }
        }
        val gworkranges = globalWorkRanges ?: listOf(0L until (args.filterIsInstance<ClBuffer>().firstOrNull()?.length?.toLong() ?: 0L))
        val global_work_offset = gworkranges.map { it.start }.toLongArray()
        val global_work_size = gworkranges.map { (it.endInclusive - it.start) + 1 }.toLongArray()
        val local_work_size = localSizes?.toLongArray()

        clEnqueueNDRangeKernel(queue.commandQueue, kernel, gworkranges.size, global_work_offset, global_work_size, local_work_size, 0, null, null)
        clReleaseKernel(kernel)
    }
}

class ClCommandQueue(val ctx: ClContext) : Closeable {
    val commandQueue = clCreateCommandQueue(ctx.context, ctx.device, 0, null)

    fun readByteBuffer(buffer: ClBuffer): ByteBuffer {
        val out = ByteBuffer.allocate(buffer.sizeInBytes).order(ByteOrder.nativeOrder())
        clEnqueueReadBuffer(commandQueue, buffer.mem, CL_TRUE, 0, buffer.sizeInBytes.toLong(), Pointer.to(out), 0, null, null)
        return out
    }

    fun waitCompleted() {
        clFinish(commandQueue)
    }

    override fun close() {
        waitCompleted()
        clReleaseCommandQueue(commandQueue)
    }

    fun ClBuffer.readInts() = readInts(this@ClCommandQueue)
    fun ClBuffer.readFloats() = readFloats(this@ClCommandQueue)
    //fun ClBuffer.readFloatsQueue() = readFloatsQueue(queue)

    operator fun ClKernel.invoke(vararg args: Any, globalWorkRanges: List<LongRange>? = null, localSizes: List<Long>? = null) = invoke(this@ClCommandQueue, *args, globalWorkRanges = globalWorkRanges, localSizes = localSizes)
    fun ClKernel.invokeQueue(vararg args: Any, globalWorkRanges: List<LongRange>? = null, localSizes: List<Long>? = null) = queue(this@ClCommandQueue, *args, globalWorkRanges = globalWorkRanges, localSizes = localSizes)

    operator inline fun <T> invoke(callback: ClCommandQueue.() -> T): T = callback(this)
    fun finish() {
        waitCompleted()
    }
}

private inline fun <T> Buffer.keepPosition(callback: () -> T): T {
    val oldPosition = position()
    return callback().apply { position(oldPosition) }
}

fun IntBuffer.toTypedArray(): IntArray = IntArray(limit()).apply { keepPosition { this@toTypedArray.get(this) } }
fun FloatBuffer.toTypedArray(): FloatArray = FloatArray(limit()).apply { keepPosition { this@toTypedArray.get(this) } }

fun IntBuffer.toList() = toTypedArray().toList()
fun FloatBuffer.toList() = toTypedArray().toList()
