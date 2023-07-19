import argparse
import torch
from transformers import AutoTokenizer, AutoModel
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="roberta-base")
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_path', type=str, default='/data/vision/oliva/datasets/BOLDMoments/prepared_data/metadata/annotations.json')
parser.add_argument("--output_path", type=str, default="./outputs/caption_features.npy")

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    with open(args.data_path, 'r') as f:
        meta_data = json.load(f)
    
    

    for i in range(0, len(questions), chunk_size):
        print("Processing chunk {} / {}".format(i, len(questions)))
        inputs = tokenizer(questions[i:i+chunk_size], return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        if i == 0:
            model_cards[model_key] = outputs.pooler_output
        else:
            model_cards[model_key] = torch.cat((model_cards[model_key], outputs.pooler_output), dim=0)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)