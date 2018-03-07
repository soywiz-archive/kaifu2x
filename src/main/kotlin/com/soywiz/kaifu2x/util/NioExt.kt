package com.soywiz.kaifu2x.util

import com.jogamp.opencl.*
import java.nio.*

fun FloatArray.toDirectBuffer(): FloatBuffer = DirectFloatBuffer(this.size).apply { put(this@toDirectBuffer); flip() }
fun IntArray.toDirectBuffer(): IntBuffer = DirectIntBuffer(this.size).apply { put(this@toDirectBuffer); flip() }

fun DirectFloatBuffer(count: Int): FloatBuffer =
    ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

fun DirectIntBuffer(count: Int): IntBuffer =
    ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()).asIntBuffer()

class CLBuffer2D<T : Buffer>(val buffer: CLBuffer<T>, val width: Int, val height: Int) {
    fun read(queue: CLCommandQueue): T {
        queue.putReadBuffer(buffer, true)
        return buffer.buffer
    }
}