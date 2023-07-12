'''

An LSP server that connects to the LLM server and APIs for doing the brainy
stuff.


USAGE:
    python lsp_server.py

'''


import sys
import logging
import uniteai.server
import uniteai.config as config
import importlib


##########
# Logging
#
# Change the basicConfig level to be the generous DEBUG, quiet the libraries,
# and allow custom loggers in modules to have their own debug levels. This
# helps especially with debugging weird concurrency quirks, ie allowing
# different processes to report things as needed, in a way that can be easily
# turned on and off per object of your attention.
#
# NOTE: Thespian has a domineering logging methodology. To customize their
#       formatter, see: `thespian.system.simpleSystemBase`.
#
#       Also, https://github.com/thespianpy/Thespian/issues/73
#
# NOTE: VSCode thinks the server is erroring if it logs to stdout from the main
#       thread, but spawned threads don't seem to have the same effect.

logging.basicConfig(
    filename='log_file.log', # TODO: feature loggers still seem to report on stdout
    # stream=sys.stdout,
    level=logging.DEBUG,
)

# Quiet the libs a little
logging.getLogger('pygls.feature_manager').setLevel(logging.WARN)
logging.getLogger('pygls.protocol').setLevel(logging.WARN)
logging.getLogger('Thespian').setLevel(logging.WARN)
logging.getLogger('asyncio').setLevel(logging.WARN)


def load_module(module_name, config_yaml, server):
    ''' Useful for loading just the modules referenced in the config. '''
    module = importlib.import_module(module_name)
    if hasattr(module, 'configure'):
        args = module.configure(config_yaml)

    if hasattr(module, 'initialize'):
        module.initialize(args, server)


def main():
    # First pass at configuration. Further passes will pick up config
    # per-feature.
    args, config_yaml, parser = config.get_args()
    server = uniteai.server.initialize()

    for module_name in args.modules:
        load_module(module_name, config_yaml, server)

    if args.tcp:
        print(f'Starting LSP on port: {args.lsp_port}')
        server.start_tcp(host='localhost', port=args.lsp_port)
    elif args.stdio:
        print(f'Starting on STDIO')
        server.start_io()


if __name__ == '__main__':
    main()
