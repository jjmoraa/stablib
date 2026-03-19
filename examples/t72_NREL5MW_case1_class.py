from pathlib import Path
from stablib import openfast
from stablib.openfast import turbine
foldername = Path('/home/jjmoraa/work/stablib/stablib/models/Land NREL 5MW 8DOF')

nrel5MW = turbine(foldername)