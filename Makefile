test_lsp:
	./test_lsp.sh

.PHONY: watch-tests
watch-tests:
	# from inotify-tools
	while find -name "*.py" ! -name '*.\#*' | inotifywait -e close_write --fromfile - ; \
		do \
			pytest --capture=no; \
		done

upload:
	rm -r dist
	python -m build
	python -m twine upload dist/*
