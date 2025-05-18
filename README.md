# [IEMDC 2025] Tutorial 8

This repository contains two example implementations of electric motors using
the open source Finite Element library [Nutils] for the hands-on part of [IEMDC
2025] Tutorial 8: Design, Modelling and Mathematical Formulations of PM-Free
Special Machines: from Theory to Practice.

The two examples are provided in the form of a [Jupyter] Notebook. The first
example, [exercise1.ipynb](exercise1.ipynb), is a simplified
reluctance motor, used to explain all concepts of modeling electric moters
using Nutils. The second example, [exercise2.ipynb](exercise2.ipynb),
contains a model of a realistic variable flux reluctance machine in space-time.

To participate either

- visit <https://iemdc.evalf.com> or
- follow the instructions below to download this repository, install Python,
  Nutils and other dependencies and launch Jupyter Notebook on your own
  computer.

## Download the contents of this repository

Download and unzip [this
archive](https://github.com/evalf/iemdc2025/archive/refs/heads/main.zip)
somewhere (we refer to this location as the *checkout dir* below). If you have
`git` you can also make a checkout of this repository.

## Installing dependencies

Make sure you have [Python]. Open a terminal and go to the *checkout dir*, the
directory where the tutorial files are located. Install [Pipenv] by running

    python3 -m pip install --user pipenv

Then run (this might take a while)

    python3 -m pipenv sync

## Starting Jupyter Notebook server

Start a Jupyter Notebook server using

    python3 -m pipenv run python3 -m notebook

from the *checkout dir*.

[Nutils]: https://nutils.org/
[IEMDC 2025]: https://www.iemdc.org/
[Python]: https://www.python.org/
[Pipenv]: https://pipenv.pypa.io/
[Jupyter]: https://jupyter.org/
