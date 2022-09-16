## Captcha breaker

Based on [vkCaptchaBreaker](https://github.com/Defasium/vkCaptchaBreaker)

Notable Changes:

+ removed mobile network
+ simplified to the working minimum
+ introduce acc metric
+ packed into tf serving

### Setup

For better experience on Windows I strongly recommend using TF via docker deployment.

Personally, I've used TF with jupyter bundled for training and testing this model.
That is also included into docker-compose

### Training

1. Create `train/` folder in repo root
2. Put your dataset in it
3. Just run `training.py`

   _Optional_
4. Run `cast_to_inference_model.py` after training to get inference model
5. Deploy via tf serving

### My results

Trained on dataset of ~1k manually labeled captcha images.

Achieved acc **>0.95**, for better results bigger dataset required.

Note: trained mostly on 4-5 symbols images, acc will drop significantly when recognizing 6-7 symbol images.
However, it should perform well in case you have big enough dataset with captcha of various size.

### Sources:

+ [vkCaptchaBreaker](https://github.com/Defasium/vkCaptchaBreaker)
+ [OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr/)

