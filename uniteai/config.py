'''

Configuration alternatively comes from a config file, or is overridden by a
command line argument.


ORDER OF PRIORITIES

1.) CLI arguments
2.) ./.uniteai.yml
3.) if not found, then read  ~/.uniteai.yml

'''

import argparse
import yaml
import os
import sys
import shutil
import logging
from uniteai.common import mk_logger

log = mk_logger('CONFIG', logging.WARN)

CONFIG_PATHS = [
    '.uniteai.yml',
    '.uniteai.yaml',
    os.path.expanduser('~/.uniteai.yaml'),
    os.path.expanduser('~/.uniteai.yml'),
]


def handle_missing_config():
    '''Possibly create a default config, and then exit.'''
    log.error('No config file found!')
    ans = input('Would you like this process to copy the default `.uniteai.yml.example` config file into the current directory?')

    if ans.lower() in ['y', 'yes']:
        log.info('''
Copying `.uniateai.yml.example` to `.uniteai.yml`
Please review it before running the LSP again.
It requires secrets (eg OpenAI Key) so you may prefer to locate it at `~/.uniteai.yml`.'''.strip())

        # New path
        config_path = '.uniteai.yml'

        # Example path
        example_config_path = '.uniteai.yml.example'

        if not os.path.exists(config_path):
            shutil.copyfile(example_config_path, config_path)
    else:
        log.error('''Please manually copy and update the file `.uniateai.yml.example` from https://github.com/freckletonj/uniteai.''')
    sys.exit(1)


def load_config(file_paths=CONFIG_PATHS):
    ''' Return first config file that exists. '''
    for file_path in file_paths:
        if os.path.exists(file_path):
            log.info(f'Reading configuration from: {file_path}')
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)

    # Will exit after possibly copying a new config
    if config_yaml is None:
        handle_missing_config()

def get_args():
    ''' This first pass will learn generic LSP-related config, and what further
    modules need to be loaded. Those modules will be able to specify their own
    configuration, which will be gathered in a second round of config parsing.
    '''
    # Load config file

    config_yaml = load_config(CONFIG_PATHS)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # LSP-related config
    parser.add_argument('--stdio', action='store_true', default=True)
    parser.add_argument('--tcp', action='store_true')
    parser.add_argument('--lsp_port', default=config_yaml.get('lsp_port', None))
    parser.add_argument('--modules', default=config_yaml.get('modules', None))

    return parser.parse_args(), config_yaml, parser
