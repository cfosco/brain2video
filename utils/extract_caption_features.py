import argparse
import torch
from transformers import AutoTokenizer, AutoModel
import json
import os
from tqdm import tqdm

attributes_format = "The event is happenning in the following scenes: %s; People are acting the following actions: %s."

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="roberta-base")
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--data_path', type=str, default='/data/vision/oliva/datasets/BOLDMoments/prepared_data/metadata/annotations.json')
parser.add_argument("--output_path", type=str, default="./outputs/")

def main(args):
    if args.model == "clip":
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cuda")
        tokenizer = clip.tokenize
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
    model.eval()

    # create output directory if not exists
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.data_path, 'r') as f:
        meta_data = json.load(f)
    
    language_features = {}

    for video_id in tqdm(meta_data.keys()):
        captions = meta_data[video_id]['text_descriptions']
        attributes = [attributes_format % (", ".join(meta_data[video_id]['scenes']), ", ".join(meta_data[video_id]['actions']))]
        input_texts = captions + attributes
        if args.model == "clip":
            inputs = tokenizer(input_texts, truncate=True).to("cuda")
            with torch.no_grad():
                text_features = model.encode_text(inputs)
                caption_features = text_features[:len(captions)]
                attribute_features = text_features[-1]
        else:
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
            outputs = model(**inputs)
            pooler_output = outputs.last_hidden_state.mean(dim=1)
            caption_features = pooler_output[:len(captions)]
            attribute_features = pooler_output[-1]
        language_features[video_id] = {
            'caption_features': caption_features.detach().cpu(),
            'attribute_features': attribute_features.detach().cpu()
        }
        print("Finish processing video %s" % video_id)

    output_filename = os.path.join(args.output_path, "language_features_model_{}.pth".format(args.model.split("/")[-1]))
    torch.save(language_features, output_filename)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)