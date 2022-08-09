import glob
import logging
import os
import re
import time
from functools import wraps
from heapq import merge

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import seaborn as sns

VERSION = "0.0.1"

DIAGONAL_LD = 1

LD_DATASET = "LD_scores"
POSITION_DATASET = "positions"
NAME_DATASET = "names"
CHUNK_PREFIX = "chunk"
START_ATTR = "start_locus"
END_ATTR = "end_locus"
PREC_ATTR = "min_score"
DEC_ATTR = "kept_decimal_places"
VERSION_ATTR = "version"
CHROMOSOME_ATTR = "chromosome"

AUX_GROUP = "aux"

MAF_COLS = [
    "Alternate_id",
    "RS_id",
    "Position",
    "Allele1",
    "Allele2",
    "MAF",
    "Minor Allele",
    "Info Score",
]
MAF_DATASET = "MAF"

# -----------------------------------------------------------
# CONVERSION FUNCTIONS
# -----------------------------------------------------------


def adjust_to_zero(sparse_matrix, precision):
    if precision:
        nonzeros = sparse_matrix.nonzero()
        nonzero_mask = np.array(np.abs(sparse_matrix[nonzeros]) < precision)[0]
        rows = nonzeros[0][nonzero_mask]
        cols = nonzeros[1][nonzero_mask]
        sparse_matrix[rows, cols] = 0

    sparse_matrix.eliminate_zeros()
    return sparse_matrix


def convert_h5(
    infile,
    outfile,
    precision=None,
    decimals=None,
    start_locus=None,
    end_locus=None,
):
    base_infile = os.path.splitext(infile)[0]

    if not start_locus or not end_locus:
        filename = base_infile.split("/")[-1]
        chromosome, start_locus, end_locus = filename.split("_")
        start_locus, end_locus = int(start_locus), int(end_locus)

    logging.debug(f"Converting {infile} loci {start_locus} to {end_locus}")

    f = h5py.File(outfile, "a")

    if len(f.keys()) == 0:  # freshly created
        f.attrs[VERSION_ATTR] = VERSION
    validate_version(f)

    group_name = f"{CHUNK_PREFIX}_{start_locus}"
    if group_name in f:
        del f[group_name]
    group = f.create_group(group_name)

    sparse_mat = sparse.tril(sparse.load_npz(base_infile + ".npz"), format="csr").T
    sparse_mat.setdiag(0)
    if decimals:
        sparse_mat.data = np.round(sparse_mat.data, decimals)
    sparse_mat = adjust_to_zero(sparse_mat, precision)

    pos_df = metadata_to_df(base_infile + ".gz")
    group.create_dataset(
        POSITION_DATASET,
        data=pos_df,
        compression="gzip",
        shape=pos_df.shape,
        dtype=pos_df.dtypes[0],
    )
    names = pos_df.index.to_numpy().astype("S")
    group.create_dataset(
        NAME_DATASET,
        data=names,
        shape=names.shape,
        dtype=names.dtype,
        compression="gzip",
    )

    group.attrs[START_ATTR] = start_locus
    group.attrs[END_ATTR] = end_locus
    group.attrs[PREC_ATTR] = precision or np.nan
    group.attrs[DEC_ATTR] = decimals or np.nan

    pos_df["relative_pos"] = np.arange(len(pos_df))
    # actually should not filter, since need for rows. instead save start and end loci for columns
    pos_df = pos_df[pos_df.BP.between(start_locus, end_locus)]

    if len(pos_df) == 0:
        logging.warning(f"No data found between loci {start_locus} and {end_locus}")
        return None

    lower_pos, upper_pos = pos_df.relative_pos[[0, -1]]
    sparse_mat = sparse_mat[lower_pos : upper_pos + 1, :]
    dense = sparse_mat.todense()
    group.create_dataset(
        LD_DATASET,
        data=dense,
        compression="gzip",
        compression_opts=9,
        shape=dense.shape,
        dtype=dense.dtype,
        scaleoffset=decimals,
    )

    f.attrs[START_ATTR] = min(f.attrs.get(START_ATTR, start_locus), start_locus)
    f.attrs[END_ATTR] = max(f.attrs.get(END_ATTR, end_locus), end_locus)


def convert_full_chromosome_h5(
    filepath, outfile, precision, decimals, start_locus, chromosome, locus_regex
):
    f = h5py.File(outfile, "a")

    f.attrs[CHROMOSOME_ATTR] = chromosome

    files = [
        (file, int(re.search(locus_regex, os.path.basename(file)).group(1)))
        for file in glob.glob(filepath)
    ]

    files.sort(key=lambda x: x[1])

    start_locus = max(start_locus, files[0][1])

    first_missing_locus = start_locus

    for i, (file, locus) in enumerate(files):
        if locus >= start_locus:
            logging.debug(f"Converting {file}")
            if i + 1 < len(files):
                next_covered_locus = files[i + 1][1]
            else:
                next_covered_locus = np.inf
            convert_h5(
                file,
                outfile,
                precision,
                decimals,
                first_missing_locus,
                next_covered_locus,
            )
            first_missing_locus = next_covered_locus

            logging.info("{:.0f}% complete".format(((i + 1) * 100) / len(files)))


def metadata_to_df(gz_file):
    df_ld_snps = pd.read_table(gz_file, sep="\s+")
    df_ld_snps.rename(
        columns={
            "chromosome": "CHR",
            "position": "BP",
            "allele1": "A1",
            "allele2": "A2",
        },
        inplace=True,
        errors="ignore",
    )
    assert "CHR" in df_ld_snps.columns
    assert "BP" in df_ld_snps.columns
    assert "A1" in df_ld_snps.columns
    assert "A2" in df_ld_snps.columns
    df_ld_snps.index = (
        df_ld_snps["CHR"].astype(str)
        + "."
        + df_ld_snps["BP"].astype(str)
        + "."
        + df_ld_snps["A1"]
        + "."
        + df_ld_snps["A2"]
    )
    return df_ld_snps[["BP"]]


def convert_maf_h5(infile, outfile):
    f = h5py.File(outfile, "a")
    group = f.require_group(AUX_GROUP)

    maf = pd.read_csv(infile, sep="\t", header=None, names=MAF_COLS)
    maf = maf.sort_values("MAF")
    maf = maf[["Position", "MAF"]].to_numpy()
    group.create_dataset(
        MAF_DATASET, data=maf, shape=maf.shape, dtype=maf.dtype, compression="gzip"
    )


# -----------------------------------------------------------
# SELECTION FUNCTIONS
# -----------------------------------------------------------


def add_slice_to_df(df, new_slice):
    if new_slice.empty:
        return df
    if df is None:
        return new_slice

    # find index overlap
    row_start, row_end = None, None
    if new_slice.index[0] in df.index:
        row_start = new_slice.index[0]
        if new_slice.index[-1] in df.index:
            row_end = new_slice.index[-1]
        else:
            row_end = df.index[-1]

    # find column overlap
    col_start, col_end = None, None
    if new_slice.columns[0] in df.columns:
        col_start = new_slice.columns[0]
        if new_slice.columns[-1] in df.columns:
            col_end = new_slice.columns[-1]
        else:
            col_end = df.columns[-1]

    if row_start and col_start:
        df.loc[row_start:row_end, col_start:col_end] = new_slice.loc[
            row_start:row_end, col_start:col_end
        ]

        right_slice = new_slice.loc[row_start:row_end, col_end:].iloc[1:, 1:]
        df = pd.concat((df, right_slice), axis=1)

        bottom_slice = new_slice.loc[row_end:, col_start:].iloc[1:, 1:]
        return pd.concat((df, bottom_slice), axis=0)

    elif row_start:
        return pd.concat((df, new_slice), axis=1)
    elif col_start:
        return pd.concat((df, new_slice), axis=0)
    else:
        return pd.concat((df, new_slice))


def add_main_slice_to_df(df, main_slice):
    if df is None:
        return main_slice
    if main_slice.index[0] not in df.index:
        return pd.concat((df, main_slice), axis=0)

    if main_slice.index[-1] in df.index:
        # everything is normal
        df.loc[
            main_slice.index[0] : main_slice.index[-1],
            main_slice.columns[0] : main_slice.columns[-1],
        ] = main_slice
    else:
        # main slice goes too far
        df.loc[main_slice.index[0] :, main_slice.columns[0] :] = main_slice.loc[
            : df.index[-1], : df.index[-1]
        ]
        df = pd.concat(
            (df, main_slice.loc[df.index[-1] :, df.index[-1] :].iloc[1:, 1:]), axis=0
        )

    return df


def subselect(df, rows, columns, range_query):
    BP_list = df[[]].copy()
    BP_list["BP"] = df.index.str.split(".").str[1].astype(int)
    BP_list["relative_pos"] = np.arange(len(BP_list))

    if range_query:
        bp_rows = BP_list[BP_list.BP.between(*rows)]
        bp_cols = BP_list[BP_list.BP.between(*columns)]
    else:
        bp_rows = BP_list[BP_list.BP.isin(rows)]
        bp_cols = BP_list[BP_list.BP.isin(columns)]

    if len(bp_rows) == 0 or len(bp_cols) == 0:
        return pd.DataFrame()

    row_inds, col_inds = bp_rows.relative_pos, bp_cols.relative_pos

    if range_query:
        return df.iloc[row_inds[0] : row_inds[-1] + 1, col_inds[0] : col_inds[-1] + 1]
    else:
        return df.iloc[row_inds, col_inds]


def extract_metadata_df_from_group(group):
    df = pd.DataFrame(
        group[POSITION_DATASET], columns=["BP"], index=group[NAME_DATASET]
    )
    df["relative_pos"] = np.arange(len(df))
    return df


def get_horizontal_slice(group, rows, columns, range_query):
    df_ld_snps = extract_metadata_df_from_group(group)
    if range_query:
        row_inds = df_ld_snps.BP.between(*rows)
        col_inds = df_ld_snps.BP.between(*columns)
    else:
        row_inds = df_ld_snps.BP.isin(rows)
        col_inds = df_ld_snps.BP.isin(columns)

    row_positions = df_ld_snps[row_inds].relative_pos
    col_positions = df_ld_snps[col_inds].relative_pos

    h_slice = None
    if len(row_positions) and len(col_positions):
        if range_query:
            h_slice = group[LD_DATASET][
                row_positions[0] : row_positions[-1] + 1,
                col_positions[0] : col_positions[-1] + 1,
            ]
        else:
            h_slice = group[LD_DATASET][row_positions.tolist()][:, col_positions]

    return pd.DataFrame(
        h_slice,
        index=row_positions.index.astype(str),
        columns=col_positions.index.astype(str),
    )


def overlap(values, interval, range_query):
    if range_query:
        overlap = max(values[0], interval[0]), min(values[1], interval[1])
        return overlap if overlap[0] <= overlap[1] else []
    return [index for index in values if interval[0] <= index < interval[1]]


def outer_overlap(i_overlap, j_overlap, range_query):
    merged = list(unique_merge((i_overlap, j_overlap)))
    return (merged[0], merged[-1]) if range_query else merged


def unique_merge(v):  # https://stackoverflow.com/a/59361748
    last = None
    for a in merge(*v):
        if a != last:
            last = a
            yield a


def get_submatrix_from_chromosome(chromosome_group, i_values, j_values, range_query):
    start_time = time.time()

    validate_version(chromosome_group)

    if not range_query:
        i_values, j_values = sorted(set(i_values)), sorted(set(j_values))

    if (
        len(i_values) == 0
        or len(j_values) == 0
        or i_values[-1] < i_values[0]
        or j_values[-1] < j_values[0]
    ):
        return pd.DataFrame()

    intervals = []
    for subgroup in chromosome_group.values():
        if subgroup.name != "/aux":
            intervals.append((subgroup.attrs[START_ATTR], subgroup.attrs[END_ATTR]))

    intervals.sort(key=lambda x: x[0])

    df = None
    for interval in intervals:
        # INTERVALS ARE ONLY FOR i VALUES!!!

        i_overlap = overlap(i_values, interval, range_query)
        j_overlap = overlap(j_values, interval, range_query)

        group = chromosome_group[f"{CHUNK_PREFIX}_{interval[0]}"]

        # get right overlap, bottom overlap, triangle overlap
        if i_overlap and j_overlap:
            full_overlap = outer_overlap(i_overlap, j_overlap, range_query)

            # triangular slice - full triangle, make symmetric, then subselect - can probably be done more efficiently
            main_slice = get_horizontal_slice(
                group, full_overlap, full_overlap, range_query
            )
            if not main_slice.empty:
                main_slice = main_slice + main_slice.T
                np.fill_diagonal(main_slice.values, DIAGONAL_LD)
                main_slice = subselect(main_slice, i_overlap, j_overlap, range_query)
                df = add_slice_to_df(df, main_slice)

            # right slice - all i in overlap, all j > interval end
        if i_overlap and j_values[-1] > interval[1]:
            right_slice = get_horizontal_slice(
                group,
                i_overlap,
                overlap(j_values, (interval[1], np.inf), range_query),
                range_query,
            )
            df = add_slice_to_df(df, right_slice)

        # bottom slice - all j in overlap, all i > interval end
        if j_overlap and i_values[-1] > interval[1]:
            bottom_slice = get_horizontal_slice(
                group,
                j_overlap,
                overlap(i_values, (interval[1], np.inf), range_query),
                range_query,
            ).T
            df = add_slice_to_df(df, bottom_slice)

    if df is None:
        df = pd.DataFrame()

    logging.debug(
        "Constructing matrix took {:.0f} seconds.".format(time.time() - start_time)
    )
    return df


def get_maf_indices_by_range(maf_dataset, lower_bound, upper_bound):
    # inclusive
    maf_values = maf_dataset[:, 1]
    start_index = np.searchsorted(maf_values, lower_bound)
    while start_index > 0 and maf_values[start_index - 1] >= lower_bound:
        start_index -= 1

    end_index = np.searchsorted(maf_values, upper_bound)
    while end_index < len(maf_values) and maf_values[end_index + 1] <= upper_bound:
        end_index += 1

    return maf_dataset[start_index:end_index, 0]


def get_submatrix_by_maf_range(chromosome_group, lower_bound, upper_bound):
    indices = get_maf_indices_by_range(
        chromosome_group[AUX_GROUP][MAF_DATASET], lower_bound, upper_bound
    )
    logging.debug(f"Found {len(indices)} matching MAFs")
    maf_result = get_submatrix_from_chromosome(
        chromosome_group, indices, indices, range_query=False
    )
    return maf_result


# -----------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------------------------------------


def plot_heatmap(df, outfile):
    logging.info("Plotting...")
    figsize = (33, 27) if outfile else (11, 9)
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, vmin=0, vmax=1, center=0)
    if outfile:
        f.savefig(outfile.split(".")[0], dpi=1000)
    else:
        plt.show()


# -----------------------------------------------------------
# METADATA FUNCTIONS
# -----------------------------------------------------------


def validate_version(f):
    existing_version = f.attrs.get(VERSION_ATTR)
    if existing_version != VERSION:
        raise ValueError(
            f"Version mismatch! Existing file is {existing_version}, but code version is {VERSION}"
        )


# -----------------------------------------------------------
# CLI WRAPPERS
# -----------------------------------------------------------


def handle_output(res, outfile, plot):
    if outfile:
        if outfile.endswith(".npz"):
            sparse.save_npz(outfile, sparse.coo_matrix(res))
        else:
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
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["warning", "info", "debug"], case_sensitive=False),
)
def cli(log_level):
    if log_level is not None:
        click.echo(f"Log level: {log_level}")
    if log_level == "warning":
        logging.basicConfig(level=logging.WARNING)
    elif log_level == "info":
        logging.basicConfig(level=logging.INFO)
    elif log_level == "debug":
        logging.basicConfig(level=logging.DEBUG)


@cli.command()
@click.argument("infile", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
@click.option("--precision", "-p", type=float, default=None)
@click.option("--decimals", "-d", type=int, default=None)
@click.option("--start-locus", "-s", type=int, default=None)
@click.option("--end-locus", "-e", type=int, default=None)
def convert(infile, outfile, precision, decimals, start_locus, end_locus):
    convert_h5(infile, outfile, precision, decimals, start_locus, end_locus)


@cli.command()
@click.argument("filepath", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
@click.option("--precision", "-p", type=float, default=None)
@click.option("--decimals", "-d", type=int, default=None)
@click.option("--start-locus", "-s", type=int, default=1)
@click.option("--chromosome", "-c", type=int, default=None)
@click.option("--locus-regex", "-r", type=str, default="_(\d+)_", show_default=True)
def convert_chromosome(
    filepath, outfile, precision, decimals, start_locus, chromosome, locus_regex
):
    logging.debug(f"Converting chromosome {chromosome}")

    convert_full_chromosome_h5(
        filepath, outfile, precision, decimals, start_locus, chromosome, locus_regex
    )


@cli.command()
@click.argument("infile", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
def convert_maf(infile, outfile):
    convert_maf_h5(infile, outfile)


@cli.command()
@click.argument("ld-file")
@click.option("--i-start", type=int, required=True)
@click.option("--i-end", type=int, required=True)
@click.option("--j-start", type=int)
@click.option("--j-end", type=int)
@output_wrapper
def submatrix(ld_file, i_start, i_end, j_start, j_end, outfile, plot):
    if j_start is None:
        logging.warning("Assuming symmetric start positions")
        j_start = i_start
    if j_end is None:
        logging.warning("Assuming symmetric end positions")
        j_end = i_end
    return get_submatrix_from_chromosome(
        h5py.File(ld_file, "r"), (i_start, i_end), (j_start, j_end), range_query=True
    )


@cli.command()
@click.argument("ld-file")
@click.option("--row-list", "-r", required=True)
@click.option("--col-list", "-c")
@output_wrapper
def submatrix_by_list(ld_file, row_list, col_list, outfile, plot):
    """
    Works with CSVs of the form:
    chr21:9411245
    chr21:9411410
    chr21:9411485
    ...
    """

    i_list = (
        pd.read_csv(row_list, header=None)
        .iloc[:, 0]
        .str.split(":")
        .str[1]
        .astype(int)
        .to_numpy()
    )

    if col_list is None:
        logging.warning("Assuming symmetric matrix")
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
@click.argument("ld-file")
@click.option("--lower-bound", "-l", type=float, default=0)
@click.option("--upper-bound", "-u", type=float, default=0.5)
@output_wrapper
def submatrix_by_maf(ld_file, lower_bound, upper_bound, outfile, plot):
    return get_submatrix_by_maf_range(h5py.File(ld_file, "r"), lower_bound, upper_bound)


if __name__ == "__main__":
    cli()
