import os
import json
import yaml
import torch
import shutil
import argparse

from utils.pdf_loader import pdf_loader
from utils.embedding import get_contents_with_embedding

def main(args):
    # load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    # init api key
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = config['API_KEY']

    # remove file
    if args.remove_file is not None:
        assert args.remove_file.endswith(".pdf"), f"Invalid PDF File: {args.remove_file}"
        file_name = args.remove_file.split('/')[-1][:-4]
        file_target_path = os.path.join(config['DATABASE']['ROOT_PATH'], file_name)
        if os.path.exists(file_target_path):
            shutil.rmtree(file_target_path)
            print(f"Delete file {file_name} from the database: Remove folder {file_target_path}")
        else:
            print(f"File does not exist in the database: {file_name}")


    # add file
    if args.add_file is not None:
        if args.add_file.endswith(".pdf"):
            all_files = [args.add_file,]
        else:
            all_files = [os.path.join(args.add_file, file) for file in os.listdir(args.add_file) if file.endswith(".pdf")]
        # load all files
        for add_file_path in all_files:
            file_name = add_file_path.split('/')[-1][:-4]
            file_source_path = add_file_path
            file_target_path = os.path.join(config['DATABASE']['ROOT_PATH'], file_name)
            if os.path.exists(file_target_path):
                print(f"Error: File {file_name}.pdf has already been imported. Duplicate imports cannot be made! Skip this time!")
            else:
                print(f"Start analyze file and add it to database. File: {file_name}")
                os.makedirs(file_target_path, exist_ok=False)
                ######################################
                # load and analyze pdf file
                raw_doc = pdf_loader(file_name, file_source_path, model_type=config['MODEL_TYPES']['PDF_ANALYZE_MODEL'])
                # save raw data
                with open(os.path.join(file_target_path, 'raw_data.json'), "w", encoding="utf-8") as f:
                    json.dump(raw_doc, f, ensure_ascii=False, indent=4)
                # get contents with embeddings
                contents_with_embed = get_contents_with_embedding(raw_doc, overlap=config['DATABASE']['OVERLAP_LENGTH'], 
                                                                           text_length=config['DATABASE']['TEXT_LENGTH'], 
                                                                           model_type=config['MODEL_TYPES']['TEXT_EMBED_MODEL'])
                # save contents with embeddings
                torch.save(contents_with_embed, os.path.join(file_target_path, "contents_with_embed.pth"))
                # for human review only
                contents_with_embed.pop('embedding')
                with open(os.path.join(file_target_path, 'contents_without_embed.json'), "w", encoding="utf-8") as f:
                    json.dump(contents_with_embed, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update RAG Database")
    parser.add_argument('--config_path', type=str, default='./configs/config.yaml', help='config path')
    parser.add_argument('--add_file', type=str, default=None, help='add pdf file path or a folder that contains pdf files')
    parser.add_argument('--remove_file', type=str, default=None, help='add pdf file path')
    args = parser.parse_args()
    main(args)