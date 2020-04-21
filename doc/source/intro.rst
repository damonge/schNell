Overview
========
Blah


Basic Structure
---------------

Blah

- a
- b
- c

Blahblig
--------

Weeee

- yes

There are a number of predefined type strings that you can see like this::

  import sacc
  print(sacc.standard_types)


You can create a type string in the correct format using the command :code:`sacc.build_data_type_name`::

  import sacc
  # the astrophysical sources involved.
  # We use 'cm21' since '21cm' starts with a number which is not allowed in variable names.
  sources = ['quasars', 'cm21']
  # the properties of these two sources we are measuring.  If they were the same
  # property for the two sources we would not repeat it
  properties = ['density', 'Intensity']
  # The statistc, Fourier space C_ell values
  statistic = 'cl'
  # There is no futher specified needed here - everything is scalar.
  subtype = None
  data_type = sacc.build_data_type_name(sources, properties, statistic, subtype)
  print(data_type)
  # prints 'quasarsCm21_densityIntensity_cl'
