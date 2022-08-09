# Efficient Storage and Querying of Linkage Disequilibrium Matrices

## Setup
Install the package with:

```
pip install ldmat
```

## Getting Started
The package includes some [example files](https://github.com/G2Lab/ldmat/tree/main/examples), 
which can be found in your venv directory under `ldmat/examples`.

The unprocessed LD matrices are too large to include here, but there is a sample 
processed matrix, `chr21_partial.h5`, which includes LD values for chromosome 21
from positions 13000001 to 22000001 (although there is no data between positions 13000001 and 14000001).
This file contains all LD values greater than 0.1, rounded to 2 decimals.

For a sample query, we can extract the square submatrix of positions 14300001 through 14400001 with the following command:
```
ldmat submatrix YOUR_VENV_DIRECTORY/ldmat/examples/chr21_partial.h5 \
--i-start 14300001 --i-end 14400001
```
This will simply print the results as a Pandas DataFrame, so you'll probably want
to save the results by adding an output file, like: `ldmat submatrix ... -o YOUR_OUTPUT.csv`

To see all the commands available, run `ldmat`.

For any specific command, you can get more information with the `--help` flag, like: `ldmat submatrix --help`.