import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import evaluate
import nltk

from PIL import Image
from transformers import AutoProcessor, AutoModelForSeq2SeqLM, AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM, pipeline, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize

def load_models():
    # load img2word model
    kosmos_name = "microsoft/kosmos-2-patch14-224"                    
    kosmos_model = AutoModelForVision2Seq.from_pretrained(kosmos_name)
    kosmos_processor = AutoProcessor.from_pretrained(kosmos_name)

    # load word2haiku model
    model_name = "fabianmmueller/deep-haiku-gpt-2"
    syllable_tokenizer = SyllableTokenizer()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False, return_tensors = "pt")

    return kosmos_model, kosmos_processor, model, tokenizer

def img2word(filename, kosmos_model, kosmos_processor):
    prompt = "<grounding>An image of"

    # User inputted image
    image = Image.open(filename)

    inputs = kosmos_processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = kosmos_model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = kosmos_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # By default, the generated  text is cleanup and the entities are extracted.
    processed_text, entities = kosmos_processor.post_process_generation(generated_text)
    # entities: list of tuples
    # [('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]

    kosmos_output = entities[0][0]
    return kosmos_output

def word2haiku(text, model, tokenizer):
    prompt = "( " + text + " = "

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    generator = pipeline('text-generation', model = "fabianmmueller/deep-haiku-gpt-2")

    result = generator(prompt)

    ### Cleaning results from pretrained model & measure perplexity in comparison to gpt2
    result_text = result[0]['generated_text']
    start = result_text.find("=")
    end = result_text.find(")")
    cleaned_result = result_text[start+1:end].strip()
    return cleaned_result

def generate_test():
    return("connect to haikumage test")