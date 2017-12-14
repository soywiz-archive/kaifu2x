package com.soywiz.kaifu2x.util

import com.soywiz.klock.TimeSpan

fun TimeSpan.toTimeString(components: Int = 3): String = toTimeString(milliseconds, components)

fun toTimeString(totalMilliseconds: Int, components: Int = 3): String {
	var timeUnit = totalMilliseconds / 1000

	if (components == 1) return "%02d".format(timeUnit)

	val seconds = timeUnit % 60
	timeUnit /= 60

	if (components == 2) return "%02d:%02d".format(timeUnit, seconds)

	val minutes = timeUnit % 60
	timeUnit /= 60

	if (components == 3) return "%02d:%02d:%02d".format(timeUnit, minutes, seconds)

	TODO("Just supported 3 components")
}
