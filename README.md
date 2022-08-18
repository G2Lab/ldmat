# Efficient Storage and Querying of Linkage Disequilibrium Matrices

## Setup
Install the package with:

```
pip install ldmat
```

## Getting Started
To see all the commands available, run `ldmat`.

For any specific command, you can get more information with the `--help` flag, like: `ldmat submatrix --help`.

The package includes some [example files](https://github.com/G2Lab/ldmat/tree/main/examples).

The unprocessed LD matrices are too large to include here, but there is a sample 
processed matrix, `chr21_partial.h5`, which includes LD values for chromosome 21
from positions 13000001 to 22000001 (although there is no data between positions 13000001 and 14000001).
This file contains all LD values greater than 0.1, rounded to 2 decimals.

### Sample Query
For a sample query, we can extract the square submatrix of positions 14300001 through 14400001 with the following command:
```
ldmat submatrix examples/chr21_partial.h5 --row-start 14300001 --row-end 14400001
```
This will simply print the results as a Pandas DataFrame, so you'll probably want
to save the results by adding an output file, like: `ldmat submatrix ... -o YOUR_OUTPUT.csv`

### Sample Compression
To try compressing a file, we've included a single LD matrix with mock data, generated from a very approximate 
Gaussian fit of the values in an actual LD matrix. In order to convert it to the compressed format, run:
```
ldmat convert-chromosome "examples/chr0_*.npz" YOUR_OUTPUT.h5 -c 0 -d 2 -m .1
```
This will find all files matching "examples/chr0_*.npz" (which is just the single provided file) and compress them,
dropping any values less than 0.1, and only keeping 2 decimals for everything. The output file can then be queried as 
described above.
