'''

Configuration alternatively comes from a config file, or is overridden by a
command line argument.


ORDER OF PRIORITIES

1.) CLI arguments
2.) ./.uniteai.yml
3.) if not found, then read  ~/.uniteai.yml

'''

import argparse
import os
import yaml
import shutil
import logging
import asyncio
from pygls.uris import from_fs_path
from pygls.server import LanguageServer
from lsprotocol.types import (
    INITIALIZE,
    INITIALIZED,
    MessageActionItem,
    MessageType,
    ShowDocumentParams,
    ShowMessageRequestParams,
    WINDOW_SHOW_DOCUMENT,
    WINDOW_SHOW_MESSAGE_REQUEST,
)
import pkg_resources
from uniteai.common import mk_logger
import sys

log = mk_logger('CONFIG', logging.INFO)

CONFIG_PATHS = [
    './.uniteai.yml',
    './.uniteai.yaml',
    os.path.expanduser('~/.uniteai.yml'),
    os.path.expanduser('~/.uniteai.yaml'),
]


def fetch_config():
    '''Entrypoint for fetching configuration.'''
    config_yaml = load_config(CONFIG_PATHS)

    # If no config was found, start the pre-config server
    if config_yaml is None:
        # Try to get config managed via pre-startup language server
        start_config_server()
        config_yaml = load_config(CONFIG_PATHS)  # Reload the config

    if config_yaml is None:
        log.error('Config couldnt be found, nor created. Please copy the example from the github repo and locate it at `~/.uniteai.yml`')
        sys.exit(0)

    # If still no config, just return None
    if config_yaml is None:
        return None, None, None

    # Process CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--stdio', action='store_true', default=True)
    parser.add_argument('--tcp', action='store_true')
    parser.add_argument('--lsp_port', default=config_yaml.get('lsp_port', None))
    parser.add_argument('--modules', default=config_yaml.get('modules', None))

    return parser.parse_args(), config_yaml, parser


def load_config(file_paths=CONFIG_PATHS):
    '''Return first config file that exists.'''
    for file_path in file_paths:
        if os.path.exists(file_path):
            log.info(f'Reading configuration from: {file_path}')
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
    return None


async def show_document(ls, uri):
    ''' Direct the client to visit a URI. '''
    log.info(f'Visiting document {uri}')
    params = ShowDocumentParams(
        uri=uri,
        external=False,  #  open the document inside the client, not externally (eg web browser)
        take_focus=True,
        selection=None
    )
    try:
        response = ls.lsp.send_request(WINDOW_SHOW_DOCUMENT, params)
    except Exception as e:
        log.error(f'Error showing document {uri}: {e}')
    return asyncio.wrap_future(response)


async def ask_user_with_message_request(ls):
    log.info('Asking user to create new config.')
    params = ShowMessageRequestParams(
        type=MessageType.Info,
        message='No config found for UniteAI. Would you like to create a default config?',
        actions=[
            MessageActionItem(title="Create in current directory"),
            MessageActionItem(title="Create in home directory"),
            MessageActionItem(title="No, I'll do it manually")
        ]
    )
    try:
        response = ls.lsp.send_request(WINDOW_SHOW_MESSAGE_REQUEST, params)
        response = await asyncio.wrap_future(response)
        if response:
            ls.show_message(f"You chose: {response.title}", MessageType.Info)
    except Exception as e:
        log.error(f'Error asking for config creation permission: {e}')
    return response


async def handle_missing_config(server):
    response = await ask_user_with_message_request(server)

    if response:
        title = response.title
        if title == "Create in current directory":
            config_path = './.uniteai.yml'
        elif title == "Create in home directory":
            config_path = os.path.expanduser('~/.uniteai.yml')
        else:
            log.error('Please manually copy and update the file `.uniateai.yml.example` from https://github.com/freckletonj/uniteai.')
            sys.exit(0)

        config_path = os.path.abspath(config_path)
        example_config_path = pkg_resources.resource_filename('uniteai', '.uniteai.yml.example')

        if not os.path.exists(config_path):
            shutil.copyfile(example_config_path, config_path)
            server.show_message(f'Config created at {config_path}', MessageType.Info)

            # visit new file
            response = await show_document(server, from_fs_path(config_path))

            # await asyncio.sleep(1)
            server.show_message(f'Restart UniteAI after editing config!', MessageType.Info)


def start_config_server():
    log.info('Starting the Config Creation Server')
    config_server = LanguageServer('config_creation_server', '0.0.0')

    @config_server.feature(INITIALIZE)
    async def initialize_callback(ls, params):
        log.info(f"Initialize Called")

    @config_server.feature(INITIALIZED)
    async def initialized_callback(ls, params):
        log.info('Config Creation Server successfully initialized')
        # Once the server is initialized, handle the missing config
        try:
            await handle_missing_config(ls)
        except Exception as e:
            log.error(f'Failed making new config: {e}')
        await ls.shutdown()
        await ls.exit()

    config_server.start_io()  # blocks
