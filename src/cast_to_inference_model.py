from tensorflow import keras
from src.model import cast_to_inference

model = keras.models.load_model('../.data/training/vk_captcha_net')

prediction_model = cast_to_inference(model)

prediction_model.save("../.data/models/vk_captcha_net/1")
