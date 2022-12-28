import sys, os
import re
from main_TANH_SUSY_layers_Epoch_width_do import train_SUSY

"""
Get the hyperparameters from the filename and the second (1) argument
"""
# Hyper-parameters
benchmark = str(sys.argv[1])

# Get the basename of the current file
string = os.path.basename(sys.argv[0])

# "layers" followed by some digits and a "_"
patt_layers = 'layers(\d+)_'
match = re.search(patt_layers, string)
layers = int(match.group(1) if match else 0)

# "Epoch" followed by some digits and a "_"
patt_Epoch = 'Epoch(\d+)_'
match = re.search(patt_Epoch, string)
epoch = int(match.group(1) if match else 20)

# "width" followed by some digits and a "_"
patt_width = 'width(\d+)_'
match = re.search(patt_width, string)
width = int(match.group(1) if match else 64)

# "do" (followed by a positive number < 1 or 
# followed by 1 ) and anything
patt_dropout = 'do(0\.\d+|1).*'
match = re.search(patt_dropout, string)
dropout = float(match.group(1) if match else .0)

print(f"layers : {layers}\nepoch : {epoch}\nwidth : {width}\n\
dropout : {dropout}\nbenchmark : {benchmark}")
# ------------------------------------------------- #

train_SUSY(layers=layers, EPOCH=epoch, width=width, dropout=dropout,
            benchmark=benchmark)

