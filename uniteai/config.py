'''

Configuration alternatively comes from a config file, or is overridden by a
command line argument.

'''

import argparse
import yaml
import os
import sys


def load_config(file_paths):
    ''' Return first config file that exists. '''
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
    raise RuntimeError("No config file found!")
    sys.exit(1)


def get_args():
    ''' This first pass will learn generic LSP-related config, and what further
    modules need to be loaded. Those modules will be able to specify their own
    configuration, which will be gathered in a second round of config parsing.
    '''
    # Load config file
    config_yaml = load_config(['.uniteai.yaml',
                               '.uniteai.yml'])

    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # LSP-related config
    parser.add_argument('--stdio', action='store_true', default=True)
    parser.add_argument('--tcp', action='store_true')
    parser.add_argument('--lsp_port', default=config_yaml.get('lsp_port', None))
    parser.add_argument('--modules', default=config_yaml.get('modules', None))

    return parser.parse_args(), config_yaml, parser
