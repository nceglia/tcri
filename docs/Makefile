# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= _build
SPHINXPRESET  ?= html
SPHINXAUTO    ?= 1
SOURCEDIR     = .
BUILDDIR      = $(SPHINXBUILD)

# Put it first so that "make" without argument is like "make help".
.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	sphinx-build -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build the documentation with auto-building enabled
html:
	sphinx-build -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# Remove build directory
clean:
	rm -rf $(BUILDDIR)

# Auto-build the documentation when files change
autobuild:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
