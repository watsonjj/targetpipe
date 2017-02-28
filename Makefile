# Makefile with some convenient quick ways to do common things

PROJECT=targetpipe

help:
	@echo ''
	@echo '$(PROJECT) available make targets:'
	@echo ''
	@echo '  help         Print this help message (the default)'
	@echo '  develop      make symlinks to this package in python install dir'
	@echo '  clean        Remove temp files'
	@echo '  test         Run tests'
	@echo ''


clean:
	$(RM) -rf build
	find . -name "*.pyc" -exec rm {} \;
	find . -name "*.so" -exec rm {} \;
	find . -name __pycache__ | xargs rm -fr

test:
	python setup.py test

pep8:
	@pep8 --statistics

trailing-spaces:
	find $(PROJECT) examples docs -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

# any other command can be passed to setup.py
%:
	python setup.py $@

