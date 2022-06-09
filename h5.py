import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import os
import pandas as pd
import click
import h5py
from heapq import merge
import re

DIAGONAL_LD = 1
FULL_MATRIX_NAME = "full"

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


def unique_merge(v):
    last = None
    for a in merge(*v):
        if a != last:
            last = a
            yield a


def sort_and_combine_lists(a, b):
    sorted_a, sorted_b = sorted(set(a)), sorted(set(b))
    return sorted_a, sorted_b, list(unique_merge((sorted_a, sorted_b)))


def combine_matrices(matrices):
    print("Combining matrices")
    columns = [column for matrix in matrices for column in matrix.columns]
    index = [index for matrix in matrices for index in matrix.index]
    return pd.DataFrame(linalg.block_diag(*matrices), columns=columns, index=index)


def adjust_to_zero(sparse_matrix, precision):
    if precision:
        nonzeros = sparse_matrix.nonzero()
        nonzero_mask = np.array(np.abs(sparse_matrix[nonzeros]) < precision)[0]
        rows = nonzeros[0][nonzero_mask]
        cols = nonzeros[1][nonzero_mask]
        sparse_matrix[rows, cols] = 0

    sparse_matrix.eliminate_zeros()
    return sparse_matrix


def convert_h5(infile, outfile, precision=0, decimals=3):
    base_infile = os.path.splitext(infile)[0]
    filename = base_infile.split("/")[-1]
    chromosome, start_snip, end_snip = filename.split("_")
    start_snip, end_snip = int(start_snip), int(end_snip)

    f = h5py.File(outfile, "a")

    group = f.require_group(f"snip_{start_snip}")

    sparse_mat = sparse.triu(sparse.load_npz(base_infile + ".npz").T, format="csr")
    sparse_mat.setdiag(0)
    sparse_mat.data = np.round(sparse_mat.data, decimals)
    sparse_mat = adjust_to_zero(sparse_mat, precision)

    pos_df = metadata_to_df(base_infile + ".gz")
    group.require_dataset(
        "positions",
        data=pos_df,
        compression="gzip",
        shape=pos_df.shape,
        dtype=pos_df.dtypes[0],
    )
    names = pos_df.index.to_numpy().astype("S")
    group.require_dataset(
        "names", data=names, shape=names.shape, dtype=names.dtype, compression="gzip"
    )

    group.attrs["start_snip"] = start_snip
    group.attrs["end_snip"] = end_snip
    group.attrs["precision"] = precision

    full = sparse_mat.todense()
    group.require_dataset(
        "full",
        data=full,
        compression="gzip",
        compression_opts=9,
        shape=full.shape,
        dtype=full.dtype,
        scaleoffset=decimals,
    )


def extract_metadata_df_from_group(group):
    df = pd.DataFrame(group["positions"], columns=["BP"], index=group["names"])
    df["relative_pos"] = np.arange(len(df))
    return df


def metadata_to_df(gz_file):
    df_ld_snps = pd.read_table(gz_file, sep="\s+")
    df_ld_snps.rename(
        columns={
            "rsid": "SNP",
            "chromosome": "CHR",
            "position": "BP",
            "allele1": "A1",
            "allele2": "A2",
        },
        inplace=True,
        errors="ignore",
    )
    assert "SNP" in df_ld_snps.columns
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


def find_overlap(a, b):
    overlap = max(a[0], b[0]), min(a[1], b[1])
    if overlap[1] < overlap[0]:
        return None
    return overlap


def get_submatrix_from_chromosome_by_range_h5(
    chromosome_group, i_start, i_end, j_start, j_end
):
    intervals = []
    for subgroup in chromosome_group.values():
        intervals.append((subgroup.attrs["start_snip"], subgroup.attrs["end_snip"]))

    intervals.sort(key=lambda x: x[0])

    submatrices = []
    for interval in intervals:
        i_overlap = find_overlap((i_start, i_end), interval)
        j_overlap = find_overlap((j_start, j_end), interval)
        if i_overlap or j_overlap:
            i_overlap = i_overlap or (-1, -1)
            j_overlap = j_overlap or (-1, -1)
            submatrices.append(
                get_submatrix_by_ranges_h5(
                    chromosome_group[f"snip_{interval[0]}"],
                    *i_overlap,
                    *j_overlap,
                )
            )
    return combine_matrices(submatrices)


def find_interval_index(val, intervals, start_index):
    index = start_index
    start_snip, end_snip = intervals[start_index]
    while val >= end_snip:
        index += 1
        if index == len(intervals):
            return None
        start_snip, end_snip = intervals[index]
    return index


def get_submatrix_from_chromosome_by_list_h5(chromosome_group, i_list, j_list):
    if len(i_list) == 0 or len(j_list) == 0:
        return pd.DataFrame()
    i_list, j_list, ind_list = sort_and_combine_lists(i_list, j_list)

    # need to find all intervals and compare

    intervals = []
    for subgroup in chromosome_group.values():
        if not subgroup.name == "/aux":
            intervals.append((subgroup.attrs["start_snip"], subgroup.attrs["end_snip"]))

    intervals.sort(key=lambda x: x[0])

    submatrices = []
    prev = 0
    interval_ind = find_interval_index(ind_list[0], intervals, 0)
    start_snip, end_snip = intervals[interval_ind]

    for start_snip, end_snip in intervals:
        # could probably be more efficient since sorted, can also remove from list
        local_is = [i for i in i_list if start_snip <= i < end_snip]
        local_js = [j for j in j_list if start_snip <= j < end_snip]
        submatrices.append(
            get_submatrix_by_indices_h5(
                chromosome_group[f"snip_{start_snip}"], local_is, local_js
            )
        )

    return combine_matrices(submatrices)


def load_symmetric_matrix_h5(group, index_df):
    if not len(index_df):
        return np.empty((0, 0))
    range_start = index_df[0]
    range_end = index_df[-1] + 1

    submatrix = group[FULL_MATRIX_NAME][range_start:range_end, range_start:range_end]
    if range_end - range_start == len(index_df):
        # we have a range
        triangular = submatrix
    else:
        ind_offsets = index_df - range_start
        triangular = submatrix[np.ix_(ind_offsets, ind_offsets)]

    symmetric = triangular + triangular.T
    np.fill_diagonal(symmetric, DIAGONAL_LD)
    return symmetric


def construct_labeled_df(
    full_matrix, df_ld_snps, i_index_df, j_index_df, combined_index_df
):
    ld_snps_ind = df_ld_snps.iloc[combined_index_df].index

    # should reduce size before creating DF
    df = pd.DataFrame(full_matrix, index=ld_snps_ind, columns=ld_snps_ind)
    return df.loc[df_ld_snps.iloc[j_index_df].index, df_ld_snps.iloc[i_index_df].index]


def get_submatrix_by_ranges_h5(group, i_start, i_end, j_start, j_end):
    df_ld_snps = extract_metadata_df_from_group(group)

    ind_temp = df_ld_snps[
        df_ld_snps.BP.between(i_start, i_end) | df_ld_snps.BP.between(j_start, j_end)
    ].relative_pos

    i_temp = df_ld_snps[df_ld_snps.BP.between(i_start, i_end)].relative_pos
    j_temp = df_ld_snps[df_ld_snps.BP.between(j_start, j_end)].relative_pos

    return construct_labeled_df(
        load_symmetric_matrix_h5(group, ind_temp), df_ld_snps, i_temp, j_temp, ind_temp
    )


def get_submatrix_by_indices_h5(group, i_list, j_list):
    i_list, j_list, ind_list = sort_and_combine_lists(i_list, j_list)
    df_ld_snps = extract_metadata_df_from_group(group)

    ind_temp = df_ld_snps[df_ld_snps.BP.isin(ind_list)].relative_pos

    i_temp = df_ld_snps[df_ld_snps.BP.isin(i_list)].relative_pos
    j_temp = df_ld_snps[df_ld_snps.BP.isin(j_list)].relative_pos
    return construct_labeled_df(
        load_symmetric_matrix_h5(group, ind_temp), df_ld_snps, i_temp, j_temp, ind_temp
    )


# def h5_searchsorted(dataset, value_index, target, how='left'):
#     lower = 0
#     upper = len(dataset)
#     while upper - lower > 1:
#         print((lower, upper, upper - lower))
#         middle = (upper + lower) // 2
#         cur_value = dataset[middle, value_index]
#         if cur_value == target:
#             upper = middle
#             lower = middle
#         elif cur_value > target:
#             upper = middle
#         else:
#             lower = middle
#
#     if how == 'left':
#         while lower > 0 and dataset[lower - 1, value_index] >= target:
#             lower -= 1
#         return lower
#     else:
#         while upper < len(dataset) and dataset[upper + 1, value_index] <= target:
#             upper += 1
#         return upper
#
# def get_maf_indices_special(maf_dataset, lower_bound, upper_bound):
#     start_index = h5_searchsorted(maf_dataset, 1, lower_bound, 'left')
#     end_index = h5_searchsorted(maf_dataset, 1, upper_bound, 'right')
#     return maf_dataset[start_index:end_index, 0]


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
    return get_submatrix_from_chromosome_by_list_h5(chromosome_group, indices, indices)


def convert_maf_h5(infile, outfile):
    f = h5py.File(outfile, "a")
    base_infile = os.path.splitext(infile)[0]
    filename = base_infile.split("/")[-1]
    chromosome = filename.split("_")[-2]
    group = f.require_group("aux")

    maf = pd.read_csv(infile, sep="\t", header=None, names=MAF_COLS)
    maf = maf.sort_values("MAF")
    maf = maf[["Position", "MAF"]].to_numpy()
    group.require_dataset(
        "MAF", data=maf, shape=maf.shape, dtype=maf.dtype, compression="gzip"
    )


# maybe make an object


@click.group()
def cli():
    pass


@cli.command()
@click.argument("ld_file")
@click.option("--i_start", type=int)
@click.option("--i_end", type=int)
@click.option("--j_start", type=int)
@click.option("--j_end", type=int)
@click.option("--outfile", "-o", default=None)
@click.option("--symmetric", "-s", is_flag=True, default=False)
def submatrix(ld_file, i_start, i_end, j_start, j_end, outfile, symmetric):
    if symmetric and (j_start is not None or j_end is not None):
        raise ValueError("Symmetric flag only compatible with i indexing.")
    if symmetric:
        j_start, j_end = i_start, i_end
    res = get_submatrix_from_chromosome_by_range_h5(
        h5py.File(ld_file, "r"), i_start, i_end, j_start, j_end
    )
    if outfile:
        # name index?
        res.to_csv(outfile)
    else:
        print(res)


@cli.command()
@click.argument("infile", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
@click.option("--precision", "-p", type=float, default=0)
@click.option("--decimals", "-d", type=int, default=3)
def convert(infile, outfile, precision, decimals):
    convert_h5(infile, outfile, precision, decimals)


@cli.command()
@click.argument("infile", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
def convert_maf(infile, outfile):
    convert_maf_h5(infile, outfile)


@cli.command()
@click.argument("ld_file")
@click.option("--lower_bound", "-l", type=float, default=0)
@click.option("--upper_bound", "-u", type=float, default=0.5)
@click.option("--outfile", "-o", default=None)
def submatrix_by_maf(ld_file, lower_bound, upper_bound, outfile):
    res = get_submatrix_by_maf_range(h5py.File(ld_file, "r"), lower_bound, upper_bound)
    if outfile:
        # name index?
        res.to_csv(outfile)
    else:
        print(res)


@cli.command()
@click.argument("directory", type=click.Path())
@click.argument("chromosome", type=int)
@click.argument("outfile", type=click.Path(exists=False))
@click.option("--precision", "-p", type=float, default=0)
@click.option("--decimals", "-d", type=int, default=3)
@click.option("--start_snip", "-s", type=int, default=1)
def convert_chromosome(directory, chromosome, outfile, precision, decimals, start_snip):

    filtered = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)) and re.match(
            f"chr{chromosome}_.*\.npz", file
        ):
            filtered.append((file, int(file.split("_")[1])))

    filtered.sort(key=lambda x: x[1])

    for (file, snip) in filtered:
        if snip >= start_snip:
            print(f"Converting {file}")
            convert_h5(os.path.join(directory, file), outfile, precision, decimals)


if __name__ == "__main__":
    cli()


# f = h5py.File('data/processed/ld_chr1.h5')
# ilist = [80232, 604807, 734460, 845283, 1302709, 1866692, 2543224, 2717973, 2739379, 2978755] + [194252273,
#  194465765,
#  195675570,
#  194384443,
#  195902318,
#  194127860,
#  194178130,
#  196699229,
#  196112944,
#  194221364]
#
# jlist = [883666, 1002539, 1076174, 1086423, 1695075, 1929834, 1950272, 2433039, 2729698, 2808238] + [195623766,
#  196781876,
#  194834457,
#  194040790,
#  196178194,
#  196086830,
#  194627363,
#  194765412,
#  194083005,
#  195917813]
# xx = get_submatrix_from_chromosome_by_list_h5(f, ilist, jlist)
