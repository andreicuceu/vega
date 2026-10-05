.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/andreicuceu/Vega/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Vega could always use more documentation, whether as part of the
official Vega docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/andreicuceu/Vega/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `Vega` for local development.

1. Fork the `Vega` repo on GitHub.
2. Clone your fork locally:

   .. code-block:: console

       git clone git@github.com:your_name_here/Vega.git
       cd Vega

3. Create and activate a dedicated environment, then install Vega and the
   development tools in editable mode:

   .. code-block:: console

       conda create --name vega-dev python=3.14
       conda activate vega-dev
       python -m pip install -e '.[dev]'

4. Install the Git hooks and validate the complete source tree:

   .. code-block:: console

       pre-commit install
       pre-commit run --all-files

   The hooks apply Ruff lint fixes and formatting. Review any modifications
   before staging them.

5. Create a branch for local development:

   .. code-block:: console

       git switch -c name-of-your-bugfix-or-feature

   Now you can make your changes locally.

6. Run focused tests while developing. Before committing, run the full local
   checks:

   .. code-block:: console

       python -m pytest
       pre-commit run --all-files

7. Commit your changes and push your branch to GitHub:

   .. code-block:: console

       git add path/to/changed/files
       git commit -m "Describe the change"
       git push -u origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. Confirm that pre-commit and the relevant local tests pass. GitHub Actions
   runs the full test matrix on Python 3.11--3.14, Ruff checks, and distribution
   validation.

Tips
----

To run a subset of tests::

    $ python -m pytest tests/test_vega.py


Dependency policy
-----------------

Vega declares its runtime dependencies without upper bounds. A lower bound appears
only where Vega's own code needs a newer interface, and ``pyproject.toml`` records
the reason next to each one (``scipy>=1.12.0`` for ``scipy.sparse.block_array`` and
``eye_array``; ``iminuit>=2.0.0`` for the ``Minuit(..., name=...)`` interface).
Because ``picca`` is a core dependency, a normal installation already resolves to
the higher floors required by ``picca`` itself (currently ``scipy>=1.16.1`` and
``iminuit>=2.26.0``). Vega does not promise support for older environments.

The CI test matrix installs the newest release of every dependency, so it is the
latest-dependencies check. Each test and documentation job uploads the exact
resolved versions as a ``freeze-python-<version>`` or ``freeze-docs`` artifact,
kept for 30 days. To reconstruct the environment of a CI run while investigating a
numerical change, download the artifact and install with:

.. code-block:: console

    python -m pip install -c freeze.txt -e '.[dev]'

Changes to a dependency lower bound should be accompanied by a comment in
``pyproject.toml`` giving the interface that requires it.

Deploying
---------

Maintainers publish releases through the manually dispatched ``Release``
workflow in GitHub Actions. First merge the release changes, including the
entry in ``HISTORY.rst``, then create and push a stable ``vX.Y.Z`` tag that is
reachable from ``master``. Run the workflow with that existing tag as its
``tag`` input.

The workflow builds and validates the wheel and source distribution, checks
their version against the tag, generates ``SHA256SUMS``, and creates the GitHub
Release. It will not create or move tags, publish to PyPI, or replace an
existing GitHub Release.
