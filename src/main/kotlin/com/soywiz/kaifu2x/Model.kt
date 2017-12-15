package com.soywiz.kaifu2x

import com.soywiz.korio.lang.DynamicAccess

data class Model(val steps: List<Step>) {
	companion object {
		fun parseJson(json: Any?): Model {
			return DynamicAccess {
				Model(json.list.map {
					Step(
						nInputPlane = it["nInputPlane"].int,
						nOutputPlane = it["nOutputPlane"].int,
						dW = it["dW"].int,
						dH = it["dH"].int,
						kW = it["kW"].int,
						kH = it["kH"].int,
						padW = it["padW"].int,
						padH = it["padH"].int,
						model_config = it["model_config"].list.map {
							ModelConfig(
								arch_name = it["arch_name"].str,
								scale_factor = it["scale_factor"].float,
								channels = it["channels"].int,
								offset = it["offset"].int
							)
						},
						weight = it["weight"].list.map {
							it.list.map {
								it.list.map { it.list }.flatMap { it }.floatArray
							}
						},
						bias = it["bias"].floatArray,
						class_name = it["class_name"].str
					)
				})
			}
		}
	}


	data class Step(
		var nInputPlane: Int = 1,
		var nOutputPlane: Int = 32,
		var dW: Int = 1,
		var dH: Int = 1,
		var kW: Int = 3,
		var kH: Int = 3,
		var padW: Int = 0,
		var padH: Int = 0,
		var model_config: List<ModelConfig> = listOf(ModelConfig()),
		var weight: List<List<FloatArray>> = listOf(listOf(FloatArray(9))),
		var bias: FloatArray = FloatArray(1),
		var class_name: String = "nn.SpatialConvolutionMM"
	)

	data class ModelConfig(
		var arch_name: String = "vgg_7",
		var scale_factor: Float = 1f,
		var channels: Int = 1,
		var offset: Int = 7
	)
}
