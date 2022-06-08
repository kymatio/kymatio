
Contributing code
=================

This document is a 'getting started' summary for contributing code, documentation, testing, and filing issues.



Feature requests/bugs: filing issues
------------------------------------
We use Github issues to track all bugs and feature requests. Feel free to
open an issue if you have found a bug or want to implement feature.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/kymatio/kymatio/issues?)
   or [pull requests](https://github.com/kymatio/kymatio/pulls).

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  Please include your operating system type and version number, as well
   as your Python, NumPy, SciPy, PyTorch, and Kymatio versions. This information
   can be found by running the following code snippet:

  ```python
  import platform; print(platform.platform())
  import sys; print("Python", sys.version)
  import numpy; print("NumPy", numpy.__version__)
  import scipy; print("SciPy", scipy.__version__)
  import torch; print("PyTorch", torch.__version__)
  import kymatio; print("Kymatio", kymatio.__version__)
  ```

- If your issue is related to GPU acceleration, please copy-paste the output
  from the [environment collection script](https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py)
  of PyTorch.
  You can get the script and run it with:
  ```python
  wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py
  # For security purposes, please check the contents of collect_env.py before running it.
  python collect_env.py
  ```

- If discussing a feature request, the community should collectively decide
whether the contribution is within the scope of the project and feasible. 
- The goal is to reach a consensus on the roadmap for the feature and its 
implementation.
- The community will decide on a series of pull requests (PRs) towards merging the 
feature. In order to make the reviewing process manageable, a decision will be 
made on how to split a large feature into multiple smaller PRs.
- Once a consensus is reached, the implementation can begin.


How to contribute
-----------------
The preferred way to contribute to Kymatio is to fork the
[main repository](http://github.com/kymatio/kymatio/) on
GitHub:

1. Fork the [project repository](http://github.com/kymatio/kymatio):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

```bash
   $ git clone git@github.com:YourLogin/kymatio.git
   $ cd kymatio
```

3. Remove any previously installed versions of Kymatio:

```bash
   $ pip uninstall kymatio
```

and install your local copy:

```bash
   $ pip install -e .
```

4. Create a branch to hold your changes:

```bash
   $ git checkout -b my-feature
```

   and start making changes. Never work in the ``master`` branch.

5. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

```bash
   $ git add modified_files
   $ git commit
```

   to record your changes in git, then push them to GitHub with:

```bash
   $ git push -u origin my-feature
```

Finally, go to the web page of the your fork of the Kymatio repository,
and click 'Pull request' to send your changes to the maintainers for
review. This will send an email to the moderators.

For further details on using Git for version control, we recommend you look
up its [documentation](http://git-scm.com/documentation).



Documentation
-------------

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the docs/ directory.
The resulting HTML files will be placed in _build/html/ and are viewable
in a web browser. See the README file in the doc/ directory for more information.

For building the documentation, you will need
[sphinx](http://sphinx.pocoo.org/),
[matplotlib](http://matplotlib.sourceforge.net/), and [numpydoc](https://pypi.python.org/pypi/numpydoc).



Acknowledgment
--------------
This document was adapted from [scikit-learn](http://scikit-learn.org/).
