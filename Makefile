test_lsp:
	./test_lsp.sh

llm_server:
	uvicorn llm_server:app --port 8000

.PHONY: watch-tests
watch-tests:
	# from inotify-tools
	while find -name "*.py" ! -name '*.\#*' | inotifywait -e close_write --fromfile - ; \
		do \
			pytest --capture=no; \
		done
