"""The numerical core of the Autoencoders add-on (see ``README.md`` in this folder for the API).

Importing this package imports nothing heavy: PyTorch is imported only inside the training worker process (``train.py`` / ``worker.py``), never in the FluoViz server.
The server-side modules (``spec``, ``forward``, ``tools``, ``diagnostics`` for everything that is not training) are pure numpy.
"""
