import pandas as pd
import click
import h5py
from functools import wraps
from h5 import (
    plot_heatmap,
    convert_h5,
    convert_full_chromsome_h5,
    convert_maf_h5,
    get_submatrix_from_chromosome,
    get_submatrix_by_maf_range,
)


# -----------------------------------------------------------
# CLI WRAPPERS
# -----------------------------------------------------------


def handle_output(res, outfile, plot):
    if outfile:
        # name index?
        res.to_csv(outfile)
    else:
        print(res)

    if plot:
        plot_heatmap(res, outfile)


def output_wrapper(function):
    function = click.option("--outfile", "-o", type=click.Path(exists=False))(function)
    function = click.option("--plot", "-p", is_flag=True, default=False)(function)

    @wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        handle_output(result, kwargs["outfile"], kwargs["plot"])

    return wrapper


@click.group()
def cli():
    pass


@cli.command()
@click.argument("infile", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
@click.option("--precision", "-p", type=float, default=0)
@click.option("--decimals", "-d", type=int, default=None)
@click.option("--start_locus", "-s", type=int, default=None)
@click.option("--end_locus", "-e", type=int, default=None)
def convert(infile, outfile, precision, decimals, start_locus, end_locus):
    convert_h5(infile, outfile, precision, decimals, start_locus, end_locus)


@cli.command()
@click.argument("directory", type=click.Path())
@click.argument("chromosome", type=int)
@click.argument("outfile", type=click.Path(exists=False))
@click.option("--precision", "-p", type=float, default=0)
@click.option("--decimals", "-d", type=int, default=None)
@click.option("--start_locus", "-s", type=int, default=1)
def convert_chromosome(
    directory, chromosome, outfile, precision, decimals, start_locus
):
    print(f"Converting chromosome {chromosome}")

    convert_full_chromsome_h5(
        directory, chromosome, outfile, precision, decimals, start_locus
    )


@cli.command()
@click.argument("infile", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
def convert_maf(infile, outfile):
    convert_maf_h5(infile, outfile)


@cli.command()
@click.argument("ld_file")
@click.option("--i_start", type=int)
@click.option("--i_end", type=int)
@click.option("--j_start", type=int)
@click.option("--j_end", type=int)
@click.option("--symmetric", "-s", is_flag=True, default=False)
@output_wrapper
def submatrix(ld_file, i_start, i_end, j_start, j_end, symmetric, outfile, plot):
    if symmetric and (j_start is not None or j_end is not None):
        raise ValueError("Symmetric flag only compatible with i indexing.")
    if symmetric:
        j_start, j_end = i_start, i_end
    return get_submatrix_from_chromosome(
        h5py.File(ld_file, "r"), (i_start, i_end), (j_start, j_end), range_query=True
    )


@cli.command()
@click.argument("ld_file")
@click.option("--row_list", "-r")
@click.option("--col_list", "-c")
@click.option("--symmetric", "-s", is_flag=True, default=False)
@output_wrapper
def submatrix_by_list(ld_file, row_list, col_list, symmetric, outfile, plot):
    """
    Works with CSVs of the form:
    chr21:9411245
    chr21:9411410
    chr21:9411485
    ...
    """

    if symmetric and col_list is not None:
        raise ValueError("Symmetric flag only compatible with row indexing.")

    i_list = (
        pd.read_csv(row_list, header=None)
        .iloc[:, 0]
        .str.split(":")
        .str[1]
        .astype(int)
        .to_numpy()
    )
    if symmetric:
        j_list = i_list
    else:
        j_list = (
            pd.read_csv(col_list, header=None)
            .iloc[:, 0]
            .str.split(":")
            .str[1]
            .astype(int)
            .to_numpy()
        )

    return get_submatrix_from_chromosome(
        h5py.File(ld_file, "r"), i_list, j_list, range_query=False
    )


@cli.command()
@click.argument("ld_file")
@click.option("--lower_bound", "-l", type=float, default=0)
@click.option("--upper_bound", "-u", type=float, default=0.5)
@output_wrapper
def submatrix_by_maf(ld_file, lower_bound, upper_bound, outfile, plot):
    return get_submatrix_by_maf_range(h5py.File(ld_file, "r"), lower_bound, upper_bound)


if __name__ == "__main__":
    cli()
