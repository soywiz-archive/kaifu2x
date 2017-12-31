package com.soywiz.kaifu2x.knum

import com.soywiz.korio.error.invalidOp
import java.nio.Buffer
import java.nio.FloatBuffer

object KNumExample {
    @JvmStatic
    fun main(args: Array<String>) {
        KNum {
            //println(floatArrayOf(1f, 2f, 3f, 4f).const)
            //println(floatArrayOf(1f, 2f, 3f, 4f).const.reshape(2, 2))
            val tensor = floatArrayOf(1f, 2f, 3f, 4f).const + floatArrayOf(4f, 5f, 6f, 7f).const
            val result = (tensor * -(1f.const)).compute().getFloatArray()
            println(result.toList())
        }
    }
}

open class KNumContext {
    class DefaultResult<T>(dims: IntArray, val _data: Buffer, type: KNum.Type) : KNum.Result<T>(dims, type) {
        override fun getData(): Buffer = _data
    }

    fun <T> compute(tensor: KNum.Tensor<T>): KNum.Result<T> {
        return when (tensor) {
            is KNum.Constant -> computeConstant(tensor)
            is KNum.Operation -> computeOperation(tensor)
            else -> invalidOp("Don't know how to compute $tensor")
        }
    }

    open fun <T> computeConstant(tensor: KNum.Constant<T>): KNum.Result<T> {
        return DefaultResult<T>(tensor.dims, tensor.data, tensor.type)
    }

    open fun <T> computeOperation(tensor: KNum.Operation<T>): KNum.Result<T> = tensor.run {
        when (op) {
            "add", "sub", "mul", "div" -> computeBinaryOp<T>(op, compute(inputs[0] as KNum.Tensor<T>), compute(inputs[1] as KNum.Tensor<T>))
            "neg" -> computeUnaryOp<T>(op, compute(inputs[0] as KNum.Tensor<T>))
            else -> invalidOp("Unsuported operation $op")
        }
    }

    open fun <T> computeUnaryOp(op: String, l: KNum.Result<T>): KNum.Result<T> {
        val of = FloatBuffer.allocate(l.numElements)
        val lf = l.getData() as FloatBuffer
        val num = l.numElements
        when (op){
            "neg" -> for (n in 0 until num) of.put(n, -lf[n])
            else -> invalidOp("Unsupported operation $op")
        }
        return DefaultResult<T>(l.dims, of, l.type)
    }

    open fun <T> computeBinaryOp(op: String, l: KNum.Result<T>, r: KNum.Result<T>): KNum.Result<T> {
        val of = FloatBuffer.allocate(l.numElements)
        val lf = l.getData() as FloatBuffer
        val rf = r.getData() as FloatBuffer
        val num = l.numElements

        fun getL(n: Int) = lf[n]
        fun getR_single(n: Int) = rf[0]
        fun getR_multi(n: Int) = rf[n]

        val fl = ::getL
        val fr = if (r.dims.size == 1 && r.dims[0] == 1) ::getR_single else ::getR_multi

        val fop: (Float, Float) -> Float = when (op) {
            "add" -> { ll, rr -> ll + rr }
            "sub" -> { ll, rr -> ll - rr }
            "mul" -> { ll, rr -> ll * rr }
            "div" -> { ll, rr -> ll / rr }
            else -> invalidOp("Unsuported operation $op")
        }

        for (n in 0 until num) of.put(n, fop(fl(n), fr(n)))
        return DefaultResult<T>(l.dims, of, l.type)
    }
}

class KNum(val ctx: KNumContext) {
    companion object {
        operator fun invoke(context: KNumContext = KNumContext(), callback: KNum.() -> Unit) {
            callback(KNum(context))
        }
    }

    enum class Type(val size: Int) { INT(4), FLOAT(4) }

    abstract class Tensor<T>(val dims: IntArray, val type: Type) {
        val numElements: Int by lazy { dims.reduce { acc, i -> acc * i } }
        override fun toString(): String = "Tensor[$type](${dims.joinToString(", ")})"
    }

    abstract class Result<T>(dims: IntArray, type: Type) : Tensor<T>(dims, type) {
        abstract fun getData(): Buffer

        fun getFloatBuffer(): FloatBuffer = getData() as FloatBuffer
        fun getFloatArray(): FloatArray = getFloatBuffer().run {
            val out = FloatArray(limit())
            position(0)
            get(out)
            out
        }
    }

    class Operation<T>(val op: String, type: Type, dims: IntArray, val inputs: Array<Tensor<*>>) : Tensor<T>(dims, type) {
        override fun toString(): String = "Operation($op[$type], ${dims.toList()})(${inputs.toList()})"
    }

    class Constant<T>(dims: IntArray, type: Type, val data: Buffer) : Tensor<T>(dims, type) {
        init {
            if (numElements != data.limit()) {
                invalidOp("${dims.toList()}")
            }
        }
    }

    val FloatArray.const: Constant<Float> get() = Constant(intArrayOf(this.size), Type.FLOAT, FloatBuffer.wrap(this))
    val Float.const: Constant<Float> get() = Constant(intArrayOf(1), Type.FLOAT, FloatBuffer.wrap(floatArrayOf(this)))

    operator fun <T> Tensor<T>.times(that: Tensor<T>): Tensor<T> = Operation<T>("mul", this.type, this.dims, arrayOf(this, that))
    operator fun <T> Tensor<T>.div(that: Tensor<T>): Tensor<T> = Operation<T>("div", this.type, this.dims, arrayOf(this, that))
    operator fun <T> Tensor<T>.plus(that: Tensor<T>): Tensor<T> = Operation<T>("add", this.type, this.dims, arrayOf(this, that))
    operator fun <T> Tensor<T>.minus(that: Tensor<T>): Tensor<T> = Operation<T>("sub", this.type, this.dims, arrayOf(this, that))
    operator fun <T> Tensor<T>.unaryMinus(): Tensor<T> = Operation<T>("neg", this.type, this.dims, arrayOf(this))

    fun <T> Tensor<T>.reshape(vararg dims: Int): Tensor<T> = Operation<T>("reshape", this.type, dims, arrayOf(this))
    fun <T> Tensor<T>.transpose(vararg axis: Int): Tensor<T> = Operation<T>("transpose", this.type, axis.map { this.dims[it] }.toIntArray(), arrayOf(this))

    fun <T> Tensor<T>.compute(): Result<T> = ctx.compute(this)
}
