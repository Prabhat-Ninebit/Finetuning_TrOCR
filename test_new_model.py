from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

# Load the model and processor
encoder_model = 'google/vit-base-patch16-224-in21k'
decoder_model = 'surajp/RoBERTa-hindi-guj-san'
trained_model_path = 'sabaridsnfuji/Hindi_Offline_Handwritten_OCR'

# Initialize the processor and model
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_model)
tokenizer = AutoTokenizer.from_pretrained(decoder_model)
processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained(trained_model_path)

# Load and preprocess the image
image_path = 'test.jpeg'
image = Image.open(image_path).convert('RGB')

# Generate text
pixel_values = processor(image, return_tensors='pt').pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Generated Text:", generated_text)
