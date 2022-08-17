import glob
import logging
import os
import re
import shutil
import time
from functools import wraps
from heapq import merge
from uuid import uuid4

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import seaborn as sns

VERSION = "0.0.2"

DIAGONAL_LD = 1

LD_DATASET = "LD_values"
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
MAF_DATASET = "MAF"

TMP_OUT = f"/tmp/ldmat_{uuid4().hex}.csv"

STREAM_THRESHOLD = 1e12

logger = logging.getLogger()

# -----------------------------------------------------------
# LOADER CLASSES
# -----------------------------------------------------------


class Loader:
    FRIENDLY_NAME = None

    def load_as_sparse_matrix(self, f):
        """
        Should return a scipy.sparse matrix, which is:
            - square
            - upper triangular
            - 0 along the diagonal
            - CSC format
        """
        raise NotImplementedError

    def load_metadata(self, f):
        """
        Should return a Pandas Dataframe, where:
            - the index contains friendly names for all positions
            - the column "BP" contains the position in the chromosome
            - there are no other columns
        """
        raise NotImplementedError

    def load_maf(self, f):
        """
        Should return a Pandas Dataframe, where:
            - the column "Position" contains the position in the chromosome
            - the column "MAF" contains the MAF value for each position
        """
        raise NotImplementedError


class BroadInstituteLoader(Loader):
    FRIENDLY_NAME = "broad-institute"
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

    def load_as_sparse_matrix(self, f):
        sparse_mat = sparse.tril(sparse.load_npz(f), format="csr").T
        sparse_mat.setdiag(0)
        return sparse_mat

    def load_metadata(self, f):
        df_ld_snps = pd.read_table(f.replace(".npz", ".gz"), sep="\s+")
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

    def load_maf(self, f):
        return pd.read_csv(f, sep="\t", header=None, names=self.MAF_COLS)


class CSVLoader(Loader):
    FRIENDLY_NAME = "csv"
    DELIMITER = ","

    def load_as_sparse_matrix(self, f):
        df = pd.read_csv(f, index_col=0, sep=self.DELIMITER).fillna(0)
        sparse_mat = sparse.triu(df, format="csc")
        sparse_mat.setdiag(0)
        return sparse_mat

    def load_metadata(self, f):
        df = pd.read_csv(f, nrows=0, index_col=0, sep=self.DELIMITER).T
        df["BP"] = df.index.str.split(".").str[1].astype(int)
        return df


class TSVLoader(CSVLoader):
    FRIENDLY_NAME = "tsv"
    DELIMITER = "\t"


# https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-name
def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


LOADER_FRIENDLY_NAMES = {cls.FRIENDLY_NAME: cls for cls in get_all_subclasses(Loader)}

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
    start_locus,
    end_locus,
    precision=None,
    decimals=None,
    loader_class=BroadInstituteLoader,
):
    loader = loader_class()
    logger.debug(f"Converting {infile} loci {start_locus} to {end_locus}")

    f = h5py.File(outfile, "a")

    if len(f.keys()) == 0:  # freshly created
        f.attrs[VERSION_ATTR] = VERSION
    validate_version(f)

    f.attrs[PREC_ATTR] = precision or np.nan
    f.attrs[DEC_ATTR] = decimals or np.nan

    group_name = f"{CHUNK_PREFIX}_{start_locus}"
    if group_name in f:
        del f[group_name]
    group = f.create_group(group_name)

    sparse_mat = loader.load_as_sparse_matrix(infile)
    if decimals:
        sparse_mat.data = np.round(sparse_mat.data, decimals)
    sparse_mat = adjust_to_zero(sparse_mat, precision)

    pos_df = loader.load_metadata(infile)
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

    pos_df["relative_pos"] = np.arange(len(pos_df))
    # actually should not filter, since need for rows. instead save start and end loci for columns
    pos_df = pos_df[pos_df.BP.between(start_locus, end_locus)]

    if len(pos_df):
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
    else:
        logger.warning(f"No data found between loci {start_locus} and {end_locus}")

    f.attrs[START_ATTR] = min(f.attrs.get(START_ATTR, start_locus), start_locus)
    f.attrs[END_ATTR] = max(f.attrs.get(END_ATTR, end_locus), end_locus)


def convert_full_chromosome_h5(
    filepath,
    outfile,
    precision,
    decimals,
    start_locus,
    chromosome,
    locus_regex,
    loader_class=BroadInstituteLoader,
):
    f = h5py.File(outfile, "a")

    f.attrs[CHROMOSOME_ATTR] = chromosome

    files = [
        (file, *map(int, re.findall(locus_regex, os.path.basename(file))))
        for file in glob.glob(filepath)
    ]

    if any(len(file) != 3 for file in files):
        raise ValueError(
            f"""Failed to find start and end loci for at least one file!
            This can usually be fixed by adjusting the locus regex (-r).
            The current locus regex is: {locus_regex}"""
        )

    files.sort(key=lambda x: x[1])

    start_locus = max(start_locus, files[0][1])

    first_missing_locus = start_locus

    for i, (file, local_start_locus, local_end_locus) in enumerate(files):
        if local_start_locus >= start_locus:
            if i + 1 < len(files):
                next_covered_locus = files[i + 1][1]
            else:
                next_covered_locus = local_end_locus
            convert_h5(
                file,
                outfile,
                first_missing_locus,
                next_covered_locus,
                precision,
                decimals,
                loader_class=loader_class,
            )
            first_missing_locus = next_covered_locus

            logger.info("{:.0f}% complete".format(((i + 1) * 100) / len(files)))


def convert_maf_h5(infile, outfile, loader_class=BroadInstituteLoader):
    loader = loader_class()

    f = h5py.File(outfile, "a")
    group = f.require_group(AUX_GROUP)

    maf = loader.load_maf(infile)
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
        if not right_slice.empty:
            df = pd.concat((df, right_slice), axis=1)

        bottom_slice = new_slice.loc[row_end:, col_start:].iloc[1:, 1:]
        if not bottom_slice.empty:
            df = pd.concat((df, bottom_slice), axis=0)
        return df

    elif row_start:
        return pd.concat((df, new_slice), axis=1)
    elif col_start:
        return pd.concat((df, new_slice), axis=0)
    else:
        return pd.concat((df, new_slice))


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


def get_submatrix_from_chromosome(
    chromosome_group, i_values, j_values, range_query, stream=None
):
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

    if stream is None:
        if range_query:
            stream = (i_values[1] - i_values[0]) * (
                j_values[1] - j_values[0]
            ) > STREAM_THRESHOLD
        else:
            stream = len(i_values) * len(j_values) > STREAM_THRESHOLD

    if stream:
        skeleton_columns = pd.Index(())
        # find full set of columns
        for interval in intervals:
            i_overlap = overlap(i_values, interval, range_query)
            j_overlap = overlap(j_values, interval, range_query)

            group = chromosome_group[f"{CHUNK_PREFIX}_{interval[0]}"]
            if i_overlap and j_overlap:
                main_slice = get_horizontal_slice(
                    group, (None, None), j_overlap, range_query
                )
                skeleton_columns = skeleton_columns.union(
                    main_slice.columns, sort=False
                )

            if i_overlap and j_values[-1] > interval[1]:
                right_slice = get_horizontal_slice(
                    group,
                    (None, None),
                    overlap(j_values, (interval[1], np.inf), range_query),
                    range_query,
                )
                skeleton_columns = skeleton_columns.union(
                    right_slice.columns, sort=False
                )

            if j_overlap and i_values[-1] > interval[1]:
                bottom_slice = get_horizontal_slice(
                    group, j_overlap, (None, None), range_query
                ).T
                skeleton_columns = skeleton_columns.union(
                    bottom_slice.columns, sort=False
                )

        pd.DataFrame(columns=skeleton_columns).to_csv(TMP_OUT)

    df = None
    for interval in intervals:
        # INTERVALS ARE ONLY FOR i VALUES!!!

        i_overlap = overlap(i_values, interval, range_query)
        j_overlap = overlap(j_values, interval, range_query)

        group = chromosome_group[f"{CHUNK_PREFIX}_{interval[0]}"]

        new_section_bottom = None
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

            new_section_bottom = len(main_slice.index)

        # right slice - all i in overlap, all j > interval end
        if i_overlap and j_values[-1] > interval[1]:
            right_slice = get_horizontal_slice(
                group,
                i_overlap,
                overlap(j_values, (interval[1], np.inf), range_query),
                range_query,
            )
            df = add_slice_to_df(df, right_slice)

            new_section_bottom = len(right_slice.index)

        # bottom slice - all j in overlap, all i > interval end
        if j_overlap and i_values[-1] > interval[1]:
            bottom_slice = get_horizontal_slice(
                group,
                j_overlap,
                overlap(i_values, (interval[1], np.inf), range_query),
                range_query,
            ).T
            df = add_slice_to_df(df, bottom_slice)
        if stream and new_section_bottom:
            write_section = df.iloc[:new_section_bottom]
            df = df.iloc[new_section_bottom:]
            write_section = write_section.reindex(columns=skeleton_columns, copy=False)
            logger.debug(f"Writing interval {interval}, {len(write_section)} rows.")
            write_section.to_csv(TMP_OUT, mode="a", header=False)

    if df is None:
        df = pd.DataFrame()

    if stream:
        df.reindex(columns=skeleton_columns, copy=False).to_csv(
            TMP_OUT, mode="a", header=False
        )
        df = TMP_OUT

    logger.debug(
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
    while end_index < len(maf_values) and maf_values[end_index] <= upper_bound:
        end_index += 1

    return maf_dataset[start_index:end_index, 0]


def get_submatrix_by_maf_range(chromosome_group, lower_bound, upper_bound, stream=None):
    indices = get_maf_indices_by_range(
        chromosome_group[AUX_GROUP][MAF_DATASET], lower_bound, upper_bound
    )
    logger.debug(f"Found {len(indices)} matching MAFs")
    return get_submatrix_from_chromosome(
        chromosome_group, indices, indices, range_query=False, stream=stream
    )


# -----------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------------------------------------


def plot_heatmap(df, outfile):
    logger.info("Plotting...")
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
    if type(res) == str:
        filepath = res
        if outfile and outfile.endswith(".csv") and not plot:
            # special handling for streaming a csv and nothing else
            shutil.move(filepath, outfile)
            return
        else:
            res = pd.read_csv(filepath, index_col=0)
            os.remove(filepath)
    if outfile:
        if outfile.endswith(".npz"):
            sparse.save_npz(outfile, sparse.coo_matrix(res))
        else:
            if not outfile.endswith(".csv"):
                logger.warning(
                    "Output file extension not understood, outputting as CSV."
                )
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


def loader_option(function):
    return click.option(
        "--loader",
        "-l",
        type=click.Choice(LOADER_FRIENDLY_NAMES.keys()),
        default=BroadInstituteLoader.FRIENDLY_NAME,
        callback=lambda ctx, param, value: LOADER_FRIENDLY_NAMES[value],
    )(function)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["warning", "info", "debug"], case_sensitive=False),
)
def cli(log_level):
    """
    A set of commands for more efficiently storing and querying linkage disequilibrium matrices.
    """
    if log_level is not None:
        click.echo(f"Log level: {log_level}")
    if log_level == "warning":
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
    elif log_level == "info":
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    elif log_level == "debug":
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")


@cli.command(short_help="compress a single file")
@click.argument("infile", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
@click.option("--min-value", "-m", type=float, default=None)
@click.option("--decimals", "-d", type=int, default=None)
@click.option("--start-locus", "-s", type=int, required=True)
@click.option("--end-locus", "-e", type=int, required=True)
@loader_option
def convert(infile, outfile, min_value, decimals, start_locus, end_locus, loader):
    convert_h5(infile, outfile, start_locus, end_locus, min_value, decimals, loader)


@cli.command(short_help="compress a bunch of files")
@click.argument("filepath", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
@click.option("--min-value", "-m", type=float, default=None)
@click.option("--decimals", "-d", type=int, default=None)
@click.option("--start-locus", "-s", type=int, default=1)
@click.option("--chromosome", "-c", type=int, required=True)
@click.option("--locus-regex", "-r", type=str, default="_(\d+)", show_default=True)
@loader_option
def convert_chromosome(
    filepath, outfile, min_value, decimals, start_locus, chromosome, locus_regex, loader
):
    logger.debug(f"Converting chromosome {chromosome}")

    convert_full_chromosome_h5(
        filepath,
        outfile,
        min_value,
        decimals,
        start_locus,
        chromosome,
        locus_regex,
        loader,
    )


@cli.command(short_help="add MAF values to an existing file")
@click.argument("infile", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
@loader_option
def convert_maf(infile, outfile, loader):
    convert_maf_h5(infile, outfile, loader)


@cli.command(short_help="select by range of positions")
@click.argument("ld-file")
@click.option("--i-start", type=int, required=True)
@click.option("--i-end", type=int, required=True)
@click.option("--j-start", type=int)
@click.option("--j-end", type=int)
@click.option("--stream/--no-stream", "-s", default=None)
@output_wrapper
def submatrix(ld_file, i_start, i_end, j_start, j_end, stream, outfile, plot):
    if j_start is None:
        logger.warning("Assuming symmetric start positions")
        j_start = i_start
    if j_end is None:
        logger.warning("Assuming symmetric end positions")
        j_end = i_end
    return get_submatrix_from_chromosome(
        h5py.File(ld_file, "r"),
        (i_start, i_end),
        (j_start, j_end),
        range_query=True,
        stream=stream,
    )


@cli.command(short_help="select by list of positions")
@click.argument("ld-file")
@click.option("--row-list", "-r", required=True)
@click.option("--col-list", "-c")
@click.option("--stream/--no-stream", "-s", default=None)
@output_wrapper
def submatrix_by_list(ld_file, row_list, col_list, stream, outfile, plot):
    """
    \b
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
        logger.warning("Assuming symmetric matrix")
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
        h5py.File(ld_file, "r"), i_list, j_list, range_query=False, stream=stream
    )


@cli.command(short_help="select by range of MAF values")
@click.argument("ld-file")
@click.option("--lower-bound", "-l", type=float, default=0)
@click.option("--upper-bound", "-u", type=float, default=0.5)
@click.option("--stream/--no-stream", "-s", default=None)
@output_wrapper
def submatrix_by_maf(ld_file, lower_bound, upper_bound, stream, outfile, plot):
    return get_submatrix_by_maf_range(
        h5py.File(ld_file, "r"), lower_bound, upper_bound, stream
    )


if __name__ == "__main__":
    cli()
