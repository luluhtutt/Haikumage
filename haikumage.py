import requests
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image
from transformers import AutoProcessor, AutoModelForSeq2SeqLM, AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM

# def img2word(filename):
