PACKAGE_NAME = levitate
BUILDDIR = build
DISTDIR = dist
DOCSDIR = docs
TESTSDIR = tests
LOGDIR = .logs

package_files := $(shell find $(PACKAGE_NAME) -name '*.py')
docs_files := $(shell find $(DOCSDIR) -name '*.rst') $(DOCSDIR)/conf.py README.rst
tests_files := $(shell find $(TESTSDIR) -name '*.py')


.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)


conda: $(LOGDIR)/conda ## Create a conda environment for development using the environment.yml file

$(LOGDIR)/conda: $(LOGDIR)
	conda env create | tee $(LOGDIR)/conda

.PHONY: clean
clean: clean_tests clean_docs clean_dist clean_install  ## Clean the build directory and some other misc files
	find . -name '*.pyc' -exec rm {} \;
	find . -name "*.DS_Store" -exec rm {} \;
	find . -name '__pycache__' -prune -exec rm -d "{}" \;
	rm -df $(BUILDDIR)

$(LOGDIR):
	@mkdir -p $(LOGDIR)

# =======
# Install
# =======

install: $(LOGDIR)/$(PACKAGE_NAME)  ## Install the package using pip, make sure that you create a virtual environment first

.PHONY: reinstall
reinstall:  ## Clears the list of installation to force a reinstall
	rm -rfd $(LOGDIR)

$(LOGDIR)/$(PACKAGE_NAME): $(LOGDIR)
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

$(BUILDDIR)/$(TESTSDIR) :
	@mkdir -p $(BUILDDIR)/$(TESTSDIR)

.PHONY: clean_tests
clean_tests :
	rm -rdf $(BUILDDIR)/$(TESTSDIR)
	rm -f .coverage
	rm -rdf .pytest_cache

$(BUILDDIR)/$(TESTSDIR)/testresult.txt : $(LOGDIR)/$(TESTSDIR) $(BUILDDIR)/$(TESTSDIR) $(tests_files) $(package_files)
	@pytest --color=yes --cache-clear --cov $(PACKAGE_NAME) --cov-report term-missing | tee $(BUILDDIR)/$(TESTSDIR)/testresult.txt

$(BUILDDIR)/$(TESTSDIR)/formatresult.txt : $(LOGDIR)/$(TESTSDIR) $(BUILDDIR)/$(TESTSDIR) $(package_files)
	$(info ============================= Formatting summary ==============================)
	@flake8 $(PACKAGE_NAME) --ignore=E501 --exit-zero > $(BUILDDIR)/$(TESTSDIR)/formatresult.txt
	@flake8 $(PACKAGE_NAME) --ignore=E501 -qq --statistics --exit-zero

$(LOGDIR)/$(TESTSDIR) : $(TESTSDIR)/requirements.txt $(LOGDIR)
	pip install -r $(TESTSDIR)/requirements.txt | tee $(LOGDIR)/$(TESTSDIR)


# =====
# Docs
# =====
.PHONY: docs pdfdocs
docs: $(BUILDDIR)/$(DOCSDIR)/html ## Build the documentation to html (recommended for reading)
pdfdocs: $(BUILDDIR)/$(DOCSDIR)/*.pdf ## Build the documentation to pdf (recommended for official deliverables)

$(BUILDDIR)/$(DOCSDIR)/html: $(LOGDIR)/$(DOCSDIR) $(docs_files) $(package_files)
	$(info ============================= Building docs ==============================)
	sphinx-build -M html $(DOCSDIR) $(BUILDDIR)/$(DOCSDIR)

$(BUILDDIR)/$(DOCSDIR)/*.pdf: $(LOGDIR)/$(DOCSDIR) $(docs_files) $(package_files)
	$(info ============================= Building docs ==============================)
	sphinx-build -M latexpdf $(DOCSDIR) $(BUILDDIR)/$(DOCSDIR)
	mv $(BUILDDIR)/$(DOCSDIR)/latex/*.pdf $(BUILDDIR)/$(DOCSDIR)

$(LOGDIR)/$(DOCSDIR) : $(DOCSDIR)/requirements.txt $(LOGDIR)
	pip install -r $(DOCSDIR)/requirements.txt | tee $(LOGDIR)/$(DOCSDIR)

.PHONY : clean_docs
clean_docs : 
	rm -rdf $(BUILDDIR)/$(DOCSDIR)