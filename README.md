# Efficient Storage and Querying of Linkage Disequilibrium Matrices

## Setup
Install the package with:

```
pip install ldmat
```

### Downloading Example Files
THIS IS A TEMPORARY WORKAROUND UNTIL A MORE PERMANENT STORAGE SOLUTION IS FOUND.

Run the following commands to create a directory called `examples` with all of the example and test files in it. 
```
mkdir ldmat_tmp
cd ldmat_tmp
git init
git config core.sparseCheckout true
git remote add origin https://github.com/G2Lab/ldmat.git
echo "examples/" > .git/info/sparse-checkout
git fetch --depth 1 origin
git pull origin main
cd ..
mv ldmat_tmp/examples .
rm -rf ldmat_tmp
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

## Public Methods
You may wish to select a submatrix without having to read to or write from the filesystem, for example if you are 
writing a script which needs to select a submatrix on the fly. For this reason, several ldmat methods have been created
specifically for public use. You can import all of these methods with `import ldmat`.

```
select_submatrix_by_range(
    ld_h5: h5py.File,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    stream: bool = None,
) -> Union[pd.DataFrame, str]
```
The inputs to this method are:
- The LD file, opened with h5py, like: `import h5py; ld_h5 = h5py.File(ld_file, "r")`.
- The start row of the range.
- The end row of the range.
- The start column of the range.
- The end column of the range.
- Whether to stream the results to a CSV. If not specified, this is decided automatically based on the expected matrix size.

```
select_submatrix_by_list(
    ld_h5: h5py.File,
    row_list: List[int],
    col_list: List[int],
    stream: bool = None,
) -> Union[pd.DataFrame, str]
```
The inputs to this method are:
- The LD file, opened with h5py, like: `import h5py; ld_h5 = h5py.File(ld_file, "r")`.
- The list of rows to include.
- The list of columns to include.
- Whether to stream the results to a CSV. If not specified, this is decided automatically based on the expected matrix size.

```
select_submatrix_by_maf(
    ld_h5: h5py.File,
    lower_bound: float,
    upper_bound: float,
    stream: bool = None,
) -> Union[pd.DataFrame, str]
```
The inputs to this method are:
- The LD file, opened with h5py, like: `import h5py; ld_h5 = h5py.File(ld_file, "r")`.
- The smallest MAF to include.
- The largest MAF to include.
- Whether to stream the results to a CSV. If not specified, this is decided automatically based on the expected matrix size.

Typically, the results of each of these methods will be a pandas DataFrame. However, if the data is streamed, the result
will be a string, the path to the CSV result.