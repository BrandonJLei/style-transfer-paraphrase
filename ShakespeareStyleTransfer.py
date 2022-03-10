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

input_sentence = input("Enter your sentence, q to quit: ")

while input_sentence != "q" and input_sentence != "quit" and input_sentence != "exit":
    greedy_decoding = paraphraser.generate(input_sentence)
    print("\ngreedy sample:\n{}\n".format(greedy_decoding))
    transferred_output = shakespeare.generate(greedy_decoding, top_p = args.top_p_value_2)
    print("\ntransferred output:\n{}\n".format(transferred_output))
    input_sentence = input("Enter your sentence, q to quit: ")


print("Exiting...")