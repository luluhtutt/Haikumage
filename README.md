# Haikumage

Turn your pictures into beautiful poetry with Haikumage!

<img height="400" alt="Haikumage UI Screenshot" src=data/ui_pic.png>

### Team Members
>* [Lulu Htutt (lh543)](https://github.com/luluhtutt)
>* [Joanna Lin (yl797)](https://github.com/Joanna-Lin-JL)

### Usage Instructions

<b>Required Libraries:</b>
* torch
* PIL
* transformers

Haikumage's user interface is implemented and deployed using Flask. Navigate to the Haikumage directory, and run the following commands:
```
export FLASK_APP=app.py
flask run
```

### Implementation
Hugging Face models used:
* [KOSMOS-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)
* [GPT2](https://huggingface.co/fabianmmueller/deep-haiku-gpt-2)

Training and evaluation of the word to haiku model can be found in the haikumage.ipynb file.