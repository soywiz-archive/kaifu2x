package com.soywiz.kaifu2x.util

import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.color.RGBA
import com.soywiz.korio.error.invalidOp
import com.soywiz.korio.util.clamp

fun Int.rgbaToYCbCr(): Int {
	val R = RGBA.getR(this)
	val G = RGBA.getG(this)
	val B = RGBA.getB(this)
	val A = RGBA.getA(this)

	val Y = (0 + (0.299 * R) + (0.587 * G) + (0.114 * B)).toInt().clamp(0, 255)
	val Cb = (128 - (0.168736 * R) - (0.331264 * G) + (0.5 * B)).toInt().clamp(0, 255)
	val Cr = (128 + (0.5 * R) - (0.418688 * G) - (0.081312 * B)).toInt().clamp(0, 255)

	return RGBA.pack(Y, Cb, Cr, A)
}

fun Int.yCbCrToRgba(): Int {
	val Y = RGBA.getR(this)
	val Cb = RGBA.getG(this)
	val Cr = RGBA.getB(this)
	val A = RGBA.getA(this)

	val R = (Y + 1.402 * (Cr - 128)).toInt().clamp(0, 255)
	val G = (Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)).toInt().clamp(0, 255)
	val B = (Y + 1.772 * (Cb - 128)).toInt().clamp(0, 255)

	return RGBA.pack(R, G, B, A)
}

fun Bitmap32.rgbaToYCbCr(): Bitmap32 = Bitmap32(width, height).apply { for (n in 0 until area) this.data[n] = this@rgbaToYCbCr.data[n].rgbaToYCbCr() }
fun Bitmap32.yCbCrToRgba(): Bitmap32 = Bitmap32(width, height).apply { for (n in 0 until area) this.data[n] = this@yCbCrToRgba.data[n].yCbCrToRgba() }

enum class ColorComponent(val index: Int) {
	RED(0), GREEN(1), BLUE(2), ALPHA(3);

	val shift = index * 8
	val clearMask = (0xFF shl shift).inv()

	fun extract(rgba: Int): Int = (rgba ushr shift) and 0xFF
	fun insert(rgba: Int, value: Int): Int = (rgba and clearMask) or ((value and 0xFF) shl shift)

	companion object {
		val ALL = values()
		operator fun get(index: Int) = ALL[index]
	}
}

val ColorComponent.Companion.Y get() = ColorComponent.RED
val ColorComponent.Companion.Cb get() = ColorComponent.GREEN
val ColorComponent.Companion.Cr get() = ColorComponent.BLUE
val ColorComponent.Companion.A get() = ColorComponent.ALPHA

fun Bitmap32.writeComponent(dstCmp: ColorComponent, from: Bitmap32, srcCmp: ColorComponent) {
	val fdata = from.data
	for (n in 0 until area) {
		data[n] = dstCmp.insert(data[n], srcCmp.extract(fdata[n]))
	}
}

fun Bitmap32.writeComponentf(dstCmp: ColorComponent, from: FloatArray2) = this.apply {
	val fdata = from.data
	for (n in 0 until area) {
		data[n] = dstCmp.insert(data[n], (fdata[n].clamp(0f, 1f) * 255).toInt())
	}
}

fun Bitmap32.readComponentf(cmp: ColorComponent, dst: FloatArray2 = FloatArray2(width, height)): FloatArray2 {
	val src = this
	val sdata = this.data
	val ddata = dst.data
	for (n in 0 until area) {
		ddata[n] = cmp.extract(sdata[n]).toFloat() / 255f
	}
	return dst
}

fun Bitmap32.get0f(): FloatArray2 = readComponentf(ColorComponent.RED)
fun Bitmap32.get1f(): FloatArray2 = readComponentf(ColorComponent.GREEN)
fun Bitmap32.get2f(): FloatArray2 = readComponentf(ColorComponent.BLUE)
fun Bitmap32.get3f(): FloatArray2 = readComponentf(ColorComponent.ALPHA)

fun Bitmap32.set0f(f: FloatArray2): Bitmap32 = writeComponentf(ColorComponent.RED, f)

fun Bitmap32.scaleNearest(sx: Int, sy: Int): Bitmap32 {
	val out = Bitmap32(width * sx, height * sy)
	for (y in 0 until out.height) {
		for (x in 0 until out.width) {
			out[x, y] = this[x / sx, y / sy]
		}
	}
	return out
}

fun ColorComponent.toStringYCbCr() = when (this.index) {
	0 -> "Y"
	1 -> "Cb"
	2 -> "Cr"
	3 -> "A"
	else -> invalidOp
}
