package com.soywiz.kaifu2x.util

import com.soywiz.korim.bitmap.Bitmap32
import com.soywiz.korim.bitmap.BitmapChannel
import com.soywiz.korio.util.clamp

fun Bitmap32.writeComponentf(dstCmp: BitmapChannel, from: FloatArray2) = this.apply {
	val fdata = from.data
	for (n in 0 until area) {
		data[n] = dstCmp.insert(data[n], (fdata[n].clamp(0f, 1f) * 255).toInt())
	}
}

fun Bitmap32.readComponentf(cmp: BitmapChannel, dst: FloatArray2 = FloatArray2(width, height)): FloatArray2 {
	val src = this
	val sdata = this.data
	val ddata = dst.data
	for (n in 0 until area) {
		ddata[n] = cmp.extract(sdata[n]).toFloat() / 255f
	}
	return dst
}
