package com.soywiz.kaifu2x.util

import com.soywiz.kmem.arraycopy
import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.bitmap.BitmapChannel
import com.soywiz.korio.util.clamp

fun Bitmap32.writeChannelf(dstCmp: BitmapChannel, from: FloatArray2) = this.apply {
	val fdata = from.data
	for (n in 0 until area) {
		data[n] = dstCmp.insert(data[n], (fdata[n].clamp(0f, 1f) * 255).toInt())
	}
}

fun Bitmap32.readChannelf(cmp: BitmapChannel, dst: FloatArray2 = FloatArray2(width, height)): FloatArray2 {
	val src = this
	val sdata = this.data
	val ddata = dst.data
	for (n in 0 until area) {
		ddata[n] = cmp.extract(sdata[n]).toFloat() / 255f
	}
	return dst
}

// @TODO: Move to Korim
fun Bitmap32.isPointInside(x: Int, y: Int) = (x >= 0 && y >= 0 && x < width && y < height)

// @TODO: Move to Korim
//fun Bitmap32.copySliceWithSizeOutOfBounds(x: Int, y: Int, width: Int, height: Int, outValue: Int = 0): Bitmap32 {
fun Bitmap32.copySliceWithSizeOutOfBounds(x: Int, y: Int, width: Int, height: Int): Bitmap32 {
	val src = this
	val dst = Bitmap32(width, height)
	val swidth = src.width
	val sheight = src.height
	for (dy in 0 until height) {
		val sy = y + dy
		if (sy in 0 until sheight) {
			var sx = x
			var cwidth = width
			var dx = 0

			//println("sx=$sx, cwidth=$cwidth, dx=$dx   [${src.width}, ${src.height}] -> [${dst.width}, ${dst.height}]")

			if (sx < 0) {
				cwidth -= sx
				dx -= sx
				sx = 0
			}
			cwidth = cwidth.clamp(0, kotlin.math.min(swidth - sx, width - dx))

			//println("   --> sx=$sx, cwidth=$cwidth, dx=$dx   [${src.width}, ${src.height}] -> [${dst.width}, ${dst.height}]")

			arraycopy(src.data, src.index(sx, sy), dst.data, dst.index(dx, dy), cwidth)
			//if (outValue != 0) {
			//	dst.data.fill(outValue, dst.index(0, dy), dst.index(dx, dy))
			//	dst.data.fill(outValue, dst.index(dx + cwidth, dy), dst.index(width, dy))
			//}
		} else {
			//if (outValue != 0) {
			//	dst.data.fill(outValue, dst.index(0, dy), dst.index(dst.width, dy))
			//}
		}
	}
	return dst
}
