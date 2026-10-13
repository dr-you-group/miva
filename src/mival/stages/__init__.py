"""Pipeline stage implementations (spec §4).

One module per stage. Modules are imported only by the stage registry in
``mival.pipeline.stage``, never by each other: stages are coupled through
files and schemas alone (spec §3.1). ``models`` imports a backend, and the
environment that has torch does not have TensorFlow, so importing this package
must not import any stage.
"""
