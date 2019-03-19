PACKAGE_NAME = levitate
BUILDDIR = build
DISTDIR = dist
DOCSDIR = docs
TESTSDIR = tests
EXAMPLESDIR = examples
LOGDIR = .logs

package_files := $(shell find $(PACKAGE_NAME) -name '*.py')
docs_files := $(shell find $(DOCSDIR) -name '*.rst') $(DOCSDIR)/conf.py README.rst
tests_files := $(shell find $(TESTSDIR) -name '*.py')
examples_files := $(shell find $(EXAMPLESDIR) -name '*.py')
examples_plots := $(examples_files:$(EXAMPLESDIR)/%.py=$(BUILDDIR)/$(EXAMPLESDIR)/%.html)

dir_guard = @mkdir -p $(@D)

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)


conda: $(LOGDIR)/conda ## Create a conda environment for development using the environment.yml file

$(LOGDIR)/conda:
	$(dir_guard)
	conda env create | tee $(LOGDIR)/conda

.PHONY: clean
clean: clean_tests clean_docs clean_dist clean_install clean_examples  ## Clean the build directory and some other misc files
	find . -name '*.pyc' -exec rm {} \;
	find . -name "*.DS_Store" -exec rm {} \;
	find . -name '__pycache__' -prune -exec rm -d "{}" \;
	find . -name "*.html" -exec rm {} \;
	rm -df $(BUILDDIR)

# =======
# Install
# =======
install: $(LOGDIR)/$(PACKAGE_NAME)  ## Install the package using pip, make sure that you create a virtual environment first

.PHONY: reinstall
reinstall:  ## Clears the list of installation to force a reinstall
	rm -rfd $(LOGDIR)

$(LOGDIR)/$(PACKAGE_NAME):
	$(dir_guard)
	pip install -e . | tee $(LOGDIR)/$(PACKAGE_NAME)

clean_install:
	rm -rdf .eggs

# ====
# Dist
# ====
.PHONY: distribute
distribute: $(BUILDDIR)/$(DISTDIR)  ## Build a distibutable package

.PHONY: upload
upload: distribute  ## Build and upload the package to pypi
	twine upload $(BUILDDIR)/$(DISTDIR)/*

.PHONY: clean_dist
clean_dist : 
	rm -rdf $(BUILDDIR)/$(DISTDIR)

$(BUILDDIR)/$(DISTDIR) : $(package_files)
	python setup.py bdist_wheel --dist-dir $(BUILDDIR)/$(DISTDIR)
	python setup.py sdist --dist-dir $(BUILDDIR)/$(DISTDIR)
	python setup.py clean --all

# =====
# Tests
# =====
.PHONY: tests
tests: $(BUILDDIR)/$(TESTSDIR)/testresult.txt $(BUILDDIR)/$(TESTSDIR)/formatresult.txt  ## Run the test suite

.PHONY: clean_tests
clean_tests :
	rm -rdf $(BUILDDIR)/$(TESTSDIR)
	rm -f .coverage
	rm -rdf .pytest_cache

$(BUILDDIR)/$(TESTSDIR)/testresult.txt : $(tests_files) $(package_files) $(LOGDIR)/$(PACKAGE_NAME)
	$(dir_guard)
	@pytest --color=yes --cache-clear --cov $(PACKAGE_NAME) --cov-report term-missing | tee $(BUILDDIR)/$(TESTSDIR)/testresult.txt

$(BUILDDIR)/$(TESTSDIR)/formatresult.txt : $(package_files) $(LOGDIR)/$(PACKAGE_NAME)
	$(dir_guard)
	$(info ============================= Formatting summary ==============================)
	@flake8 $(PACKAGE_NAME) --ignore=E501 --exit-zero > $(BUILDDIR)/$(TESTSDIR)/formatresult.txt
	@flake8 $(PACKAGE_NAME) --ignore=E501 -qq --statistics --exit-zero


# ====
# Docs
# ====
.PHONY: docs pdfdocs
docs: $(BUILDDIR)/$(DOCSDIR)/html ## Build the documentation to html (recommended for reading)
pdfdocs: $(BUILDDIR)/$(DOCSDIR)/*.pdf ## Build the documentation to pdf (recommended for official deliverables)

$(BUILDDIR)/$(DOCSDIR)/html: $(docs_files) $(package_files) $(examples_files) $(examples_plots) $(LOGDIR)/$(PACKAGE_NAME)
	$(info ============================= Building docs ==============================)
	sphinx-build -M html $(DOCSDIR) $(BUILDDIR)/$(DOCSDIR)

$(BUILDDIR)/$(DOCSDIR)/*.pdf: $(docs_files) $(package_files) $(examples_files) $(LOGDIR)/$(PACKAGE_NAME)
	$(info ============================= Building docs ==============================)
	sphinx-build -M latexpdf $(DOCSDIR) $(BUILDDIR)/$(DOCSDIR)
	mv $(BUILDDIR)/$(DOCSDIR)/latex/*.pdf $(BUILDDIR)/$(DOCSDIR)

.PHONY : clean_docs
clean_docs : 
	rm -rdf $(BUILDDIR)/$(DOCSDIR)


# ========
# Examples
# ========
.PHONY: examples
examples: $(examples_plots)  ## Runs the example scripts and plots the output.

$(BUILDDIR)/$(EXAMPLESDIR)/%.html: $(EXAMPLESDIR)/%.py
	$(dir_guard)
	python $<
	mv $(shell echo $@ | sed -E 's/$(BUILDDIR)\/$(EXAMPLESDIR)\/(.*)/\1/') $@

.PHONY: clean_examples
clean_examples:
	rm -rdf $(BUILDDIR)/$(EXAMPLESDIR)
