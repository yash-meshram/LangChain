# Transformers
# it will download the default model for 'sentiment-analysis' when we first run it
# downloaded model saved in "c:/user/.cache/huggingface..."
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
# analyse the sentiment of below sentence
response = classifier('Hugging face is good.')
print(response)
# response will be 'positive' and 'negative' label and score


# downloading the model
from huggingface_hub import snapshot_download
# need access token for Llama model - submitted the request - waiting for response
snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")


# transformers
# easy access to pre-train transformer models
from transformers import pipeline
qa = pipeline("question-answering")
qa(question="What is Hugging Face?", context="Hugging Face is a company that provides tools for NLP.")


# datasets
# load and preprocess dataset easily for ML/NLP tasks
from datasets import load_dataset
dataset = load_dataset('imdb')
print(dataset['train'][0])


# evaluate
# easy-to-use evaluation metrics for ML model
import evaluate
accuracy = evaluate.load('accuracy')
results = accuracy.compute(
    predictions = [0, 1, 1],
    references = [1, 0, 1]
)
print(results)


# accelerate
# simplifies multi-GPU, TPU and mixed-precision training
# Automatically handles device placement (CPU/GPU/TPU)
# Simplifies distributed training (DataParallel, DDP)
# Abstracts away complex setup for scalable training
from accelerate import Accelerator
model, optimizer, dataloader = Accelerator(
    model, optimizer, dataloader
)


# timm
# provide a large collection of pre-train computer vision model
import timm
import torch
model = timm.create_model('resnet50', pretrained = True)
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(y)



# | Library        | Role                                    |
# | -------------- | --------------------------------------- |
# | `transformers` | Provides NLP & vision models            |
# | `datasets`     | Supplies datasets for training/testing  |
# | `evaluate`     | Calculates performance metrics          |
# | `accelerate`   | Boosts training performance (multi-GPU) |
# | `timm`         | Supplies advanced CV models             |


