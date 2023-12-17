from angorapy.common.postprocessors import StateNormalizer
import keras


def normalization_layer_from_postprocessor(modality: str, postprocessor: StateNormalizer):
    normalization_layer = keras.layers.Normalization(
        mean=postprocessor.mean,
        variance=postprocessor.variance,
        name=postprocessor.name
    )

    return normalization_layer