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
from pygls import lsp
from pygls.uris import from_fs_path
from pygls.server import LanguageServer
from lsprotocol.types import (
    MessageType,
    MessageActionItem,
    ShowMessageRequestParams,
    TEXT_DOCUMENT_DID_CLOSE,
    DidCloseTextDocumentParams,
    INITIALIZE,
    INITIALIZED,
    WINDOW_SHOW_MESSAGE_REQUEST,
    WINDOW_SHOW_DOCUMENT,
    ShowDocumentParams,
    MessageType,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_SAVE,
    DidSaveTextDocumentParams,
    TEXT_DOCUMENT_DID_CHANGE,
    DidChangeTextDocumentParams,
    WORKSPACE_DID_CHANGE_WORKSPACE_FOLDERS,
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


# log = mk_logger('CONFIG', logging.INFO)

# CONFIG_PATHS = [
#     './.uniteai.yml',
#     './.uniteai.yaml',
#     os.path.expanduser('~/.uniteai.yaml'),
#     os.path.expanduser('~/.uniteai.yml'),
# ]


# ##################################################
# # Main entrypoint to get config


# async def get_args(server):
#     ''' The entrypoint to configuration. This will launch a "config LSP server"
#     to handle config creation if it doesn't already exist.'''

#     # Load config file
#     config_yaml = load_config(CONFIG_PATHS)

#     if config_yaml is None:
#         config_yaml = await handle_missing_config(server)

#         if not config_yaml:  # In case the user decides not to create a default config
#             return None, None, None

#     # Rest of your code as is
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--stdio', action='store_true', default=True)
#     parser.add_argument('--tcp', action='store_true')
#     parser.add_argument('--lsp_port', default=config_yaml.get('lsp_port', None))
#     parser.add_argument('--modules', default=config_yaml.get('modules', None))

#     return parser.parse_args(), config_yaml, parser


# def load_config(file_paths=CONFIG_PATHS):
#     ''' Return first config file that exists. '''
#     for file_path in file_paths:
#         file_path = os.path.abspath(file_path)
#         if os.path.exists(file_path):
#             log.info(f'Reading configuration from: {file_path}')
#             with open(file_path, 'r') as f:
#                 return yaml.safe_load(f)
#     return None


# ##################################################
# # Config Server

# async def ask_user_with_message_request(server):
#     params = ShowMessageRequestParams(
#         type=MessageType.Info,
#         message='No config found for UniteAI. Would you like to create a default config?',
#         actions=[
#             MessageActionItem(title="Create in current directory"),
#             MessageActionItem(title="Create in home directory"),
#             MessageActionItem(title="No, I'll do it manually")
#         ]
#     )
#     response_future = server.lsp.send_request(lsp.WINDOW_SHOW_MESSAGE_REQUEST, params)
#     response = await asyncio.wrap_future(response_future)
#     return response


# async def handle_missing_config(server):
#     '''Possibly create a default config, and then exit.'''
#     log.error('No config file found!')

#     response = await ask_user_with_message_request(server)

#     if not response:  # If no response was provided
#         log.error('Please manually copy and update the file `.uniateai.yml.example` from https://github.com/freckletonj/uniteai.')
#         return None

#     title = response.title

#     if title == "Create in current directory":
#         config_path = '.uniteai.yml'
#     elif title == "Create in home directory":
#         config_path = os.path.expanduser('~/.uniteai.yml')
#     else:
#         log.error('Please manually copy and update the file `.uniateai.yml.example` from https://github.com/freckletonj/uniteai.')
#         return None

#     example_config_path = pkg_resources.resource_filename('uniteai', '.uniteai.yml.example')

#     if not os.path.exists(config_path):
#         shutil.copyfile(example_config_path, config_path)

#         # Informing the user about the new config and restarting the LSP or editor
#         server.show_message(f"New config created at {config_path}. Please review and possibly restart the LSP or editor for changes to take effect.", MessageType.Info)

#         # It seems pygls doesn't have a direct method to open files. This might be a client-specific command.
#         # You can instruct the user to open it or research if your client has a specific command for it.

#     return None


# def start_config_server():
#     config_server = lsp.LanguageServer()

#     @config_server.feature(TEXT_DOCUMENT_DID_CLOSE)
#     async def close_document(ls, params: DidCloseTextDocumentParams):
#         # If the closed document is the configuration file
#         if params.textDocument.uri == "uri_of_the_config_file":
#             await config_server.shutdown()
#             config_server.exit()

#     logging.info('Starting pre-config server on STDIO to handle missing configuration')
#     config_server.start_io()

#     # You could either wait for a certain message or command from the client
#     # indicating that the config process is done or add some other way to
#     # detect when to stop this server and start the main server.

#     return config_server




##################################################

# import argparse
# import yaml
# import os
# import sys
# import shutil
# import logging
# from uniteai.common import mk_logger
# import pkg_resources

# log = mk_logger('CONFIG', logging.INFO)

# CONFIG_PATHS = [
#     './.uniteai.yml',
#     './.uniteai.yaml',
#     os.path.expanduser('~/.uniteai.yaml'),
#     os.path.expanduser('~/.uniteai.yml'),
# ]


# def handle_missing_config():
#     '''Possibly create a default config, and then exit.'''
#     log.error('No config file found!')
#     ans = input('Would you like this process to copy the default `.uniteai.yml.example` config file into the current directory? (y)es / (n)o')

#     if ans.lower() in ['y', 'yes']:
#         log.info('''
# Copying `.uniateai.yml.example` to `.uniteai.yml`
# Please review it before running the LSP again.
# It requires secrets (eg OpenAI Key) so you may prefer to locate it at `~/.uniteai.yml`.'''.strip())

#         # New path
#         config_path = '.uniteai.yml'

#         # Example path
#         example_config_path = pkg_resources.resource_filename('uniteai', '.uniteai.yml.example')

#         if not os.path.exists(config_path):
#             shutil.copyfile(example_config_path, config_path)
#     else:
#         log.error('''Please manually copy and update the file `.uniateai.yml.example` from https://github.com/freckletonj/uniteai.''')
#     sys.exit(1)


# def load_config(file_paths=CONFIG_PATHS):
#     ''' Return first config file that exists. '''
#     for file_path in file_paths:
#         file_path = os.path.abspath(file_path)
#         if os.path.exists(file_path):
#             log.info(f'Reading configuration from: {file_path}')
#             with open(file_path, 'r') as f:
#                 return yaml.safe_load(f)


# def get_args():
#     ''' This first pass will learn generic LSP-related config, and what further
#     modules need to be loaded. Those modules will be able to specify their own
#     configuration, which will be gathered in a second round of config parsing.
#     '''
#     # Load config file
#     config_yaml = load_config(CONFIG_PATHS)

#     if config_yaml is None:
#         # Will exit after possibly copying a new config
#         handle_missing_config()

#     # Parse command-line arguments
#     parser = argparse.ArgumentParser()

#     # LSP-related config
#     parser.add_argument('--stdio', action='store_true', default=True)
#     parser.add_argument('--tcp', action='store_true')
#     parser.add_argument('--lsp_port', default=config_yaml.get('lsp_port', None))
#     parser.add_argument('--modules', default=config_yaml.get('modules', None))

#     return parser.parse_args(), config_yaml, parser
