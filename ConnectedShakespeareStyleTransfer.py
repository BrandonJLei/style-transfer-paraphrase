import argparse
import logging
import sys
import torch
from style_paraphrase.inference_utils import GPT2Generator

    
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default="", type=str)
parser.add_argument('--top_p_value_1', default=0.1, type=float)
parser.add_argument('--top_p_value_2', default=0.9, type=float)
args = parser.parse_args()

if not torch.cuda.is_available():
    print("Please check if a GPU is available or your Pytorch installation is correct.")
    sys.exit()

print("Loading paraphraser...")
paraphraser = GPT2Generator(args.model_dir + "paraphraser_gpt2_large", upper_length="same_5")

print("Loading Shakespeare model...")
shakespeare = GPT2Generator(args.model_dir + "model_299")

print("\n\nNOTE: Ignore the weight mismatch error, this is due to different huggingface/transformer versions + minor modifications I did myself, shouldn't affect the paraphrases.\n\n")

#input_sentence = input("Enter your sentence, q to quit: ")
f = open("GeneratedText.txt", "r")
text = f.read()
f.close()
print(text)
paraphrased_text = ""
transfered_text = ""
sentences = text.split(". ")
for i in sentences:
    decoding = paraphraser.generate(i)
    #print("\ngreedy sample:\n{}\n".format(decoding))
    paraphrased_text += decoding
    transferred_output = shakespeare.generate(decoding, top_p = args.top_p_value_2)
    transfered_text += transferred_output
    #print("\ntransferred output:\n{}\n".format(transferred_output))

print("Paraphrased: " + paraphrased_text + "\n")
print("Style-transfered: " + transfered_text + "\n")

print("Exiting...")
