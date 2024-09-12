import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    # prompt related
    parser.add_argument("--prompt", type=str, help="a single text prompt to generate an image. (if `--from_file` is specified, this will be ignored.)")
    parser.add_argument("--prompt_from_file", type=str, help="if specified, load prompts from this file, separated by json elements")

    parser.add_argument("--config_from_file", type=str, default="configs/config.json", help="if specified, load config from this file, separated by json elements")
    
    opt = parser.parse_args()
    return opt