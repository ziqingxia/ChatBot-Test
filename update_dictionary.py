import os
import json
import yaml
import torch
import shutil
import argparse

from utils.embedding import get_dictionary_with_embedding

def main(args):
    # load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    # init api key
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = config['API_KEY']

    # remove file
    if args.remove_file is not None:
        assert args.remove_file.endswith(".json"), f"Invalid Json File: {args.remove_file}"
        file_name = args.remove_file.split('/')[-1][:-5]
        file_target_path = os.path.join(config['DICTIONARY']['ROOT_PATH'], file_name)
        if os.path.exists(file_target_path):
            shutil.rmtree(file_target_path)
            print(f"Delete file {file_name} from the dictionary: Remove folder {file_target_path}")
        else:
            print(f"Error: File does not exist in the dictionary: {file_name}")


    # add file
    if args.add_file is not None:
        if args.add_file.endswith(".json"):
            all_files = [args.add_file,]
        else:
            all_files = [os.path.join(args.add_file, file) for file in os.listdir(args.add_file) if file.endswith(".json")]
        # load all files
        for add_file_path in all_files:
            file_name = add_file_path.split('/')[-1][:-5]
            file_source_path = add_file_path
            file_target_path = os.path.join(config['DICTIONARY']['ROOT_PATH'], file_name)
            if os.path.exists(file_target_path):
                print(f"SKIP: File {file_name}.json has already been imported. Duplicate imports cannot be made! Skip this time!")
            else:
                print(f"Start add json file to dictionary list. File: {file_name}")
                os.makedirs(file_target_path, exist_ok=False)
                ######################################
                # load dictionary file
                raw_dict = json.load(open(file_source_path))
                # save raw data
                with open(os.path.join(file_target_path, 'raw_dict.json'), "w", encoding="utf-8") as f:
                    json.dump(raw_dict, f, ensure_ascii=False, indent=4)
                # get contents with embeddings
                contents_with_embed = get_dictionary_with_embedding(raw_dict, file_name, model_type=config['MODEL_TYPES']['TEXT_EMBED_MODEL'])

                # save contents with embeddings
                torch.save(contents_with_embed, os.path.join(file_target_path, "contents_with_embed.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update RAG Json Dictionary")
    parser.add_argument('--config_path', type=str, default='./configs/config.yaml', help='config path')
    parser.add_argument('--add_file', type=str, default=None, help='add json dictionary path or a folder that contains json dictionary files')
    parser.add_argument('--remove_file', type=str, default=None, help='add json dictionary file path')
    args = parser.parse_args()
    main(args)