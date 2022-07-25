import numpy as np
import scipy.sparse as sparse
import os
import pandas as pd
import click
import h5py
from heapq import merge
import seaborn as sns
import matplotlib.pyplot as plt
from functools import wraps

DIAGONAL_LD = 1

LD_DATASET = "LD_scores"
POSITION_DATASET = "positions"
NAME_DATASET = "names"
CHUNK_PREFIX = "chunk"
START_ATTR = "start_locus"
END_ATTR = "end_locus"
PREC_ATTR = "precision"

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
    precision=0,
    decimals=None,
    start_locus=None,
    end_locus=None,
):
    base_infile = os.path.splitext(infile)[0]

    if not start_locus or not end_locus:
        filename = base_infile.split("/")[-1]
        chromosome, start_locus, end_locus = filename.split("_")
        start_locus, end_locus = int(start_locus), int(end_locus)

    print(f"Converting {infile} loci {start_locus} to {end_locus}")

    f = h5py.File(outfile, "a")

    group = f.require_group(f"{CHUNK_PREFIX}_{start_locus}")

    sparse_mat = sparse.triu(sparse.load_npz(base_infile + ".npz").T, format="csr")
    sparse_mat.setdiag(0)
    if decimals:
        sparse_mat.data = np.round(sparse_mat.data, decimals)
    sparse_mat = adjust_to_zero(sparse_mat, precision)

    pos_df = metadata_to_df(base_infile + ".gz")
    group.require_dataset(
        POSITION_DATASET,
        data=pos_df,
        compression="gzip",
        shape=pos_df.shape,
        dtype=pos_df.dtypes[0],
    )
    names = pos_df.index.to_numpy().astype("S")
    group.require_dataset(
        NAME_DATASET,
        data=names,
        shape=names.shape,
        dtype=names.dtype,
        compression="gzip",
    )

    group.attrs[START_ATTR] = start_locus
    group.attrs[END_ATTR] = end_locus
    group.attrs[PREC_ATTR] = precision

    pos_df["relative_pos"] = np.arange(len(pos_df))
    # actually should not filter, since need for rows. instead save start and end loci for columns
    pos_df = pos_df[pos_df.BP.between(start_locus, end_locus)]

    if len(pos_df) == 0:
        print(f"No data found between loci {start_locus} and {end_locus}")
        return None

    lower_pos, upper_pos = pos_df.relative_pos[[0, -1]]
    sparse_mat = sparse_mat[lower_pos : upper_pos + 1, :]
    dense = sparse_mat.todense()
    group.require_dataset(
        LD_DATASET,
        data=dense,
        compression="gzip",
        compression_opts=9,
        shape=dense.shape,
        dtype=dense.dtype,
        scaleoffset=decimals,
    )


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
    group = f.require_group("aux")

    maf = pd.read_csv(infile, sep="\t", header=None, names=MAF_COLS)
    maf = maf.sort_values("MAF")
    maf = maf[["Position", "MAF"]].to_numpy()
    group.require_dataset(
        "MAF", data=maf, shape=maf.shape, dtype=maf.dtype, compression="gzip"
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
        chromosome_group["aux"]["MAF"], lower_bound, upper_bound
    )
    print(f"Found {len(indices)} matching MAFs")
    maf_result = get_submatrix_from_chromosome(
        chromosome_group, indices, indices, range_query=False
    )
    assert all(np.diagonal(maf_result) == 1)
    return maf_result


# -----------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------------------------------------


def plot_heatmap(df, outfile):
    print("Plotting...")
    figsize = (33, 27) if outfile else (11, 9)
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, vmin=0, vmax=1, center=0)
    if outfile:
        f.savefig(outfile.split(".")[0], dpi=1000)
    else:
        plt.show()


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

    filtered = []
    for file in os.listdir(directory):
        if (
            os.path.isfile(os.path.join(directory, file))
            and file.startswith(f"chr{chromosome}_")
            and file.endswith(".npz")
        ):
            filtered.append((file, int(file.split("_")[1])))

    filtered.sort(key=lambda x: x[1])

    start_locus = max(start_locus, filtered[0][1])

    first_missing_locus = start_locus

    for i, (file, locus) in enumerate(filtered):
        if locus >= start_locus:
            print(f"Converting {file}")
            if i + 1 < len(filtered):
                next_covered_locus = filtered[i + 1][1]
            else:
                next_covered_locus = np.inf
            convert_h5(
                os.path.join(directory, file),
                outfile,
                precision,
                decimals,
                first_missing_locus,
                next_covered_locus,
            )
            first_missing_locus = next_covered_locus

            print("{:.2f}% complete".format(((i + 1) * 100) / len(filtered)))


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
