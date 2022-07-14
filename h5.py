import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import os
import pandas as pd
import click
import h5py
from heapq import merge

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


def convert_h5(
    infile, outfile, precision=0, decimals=3, start_snip=None, end_snip=None
):
    base_infile = os.path.splitext(infile)[0]

    if not start_snip or not end_snip:
        filename = base_infile.split("/")[-1]
        chromosome, start_snip, end_snip = filename.split("_")
        start_snip, end_snip = int(start_snip), int(end_snip)

    print(f"Converting {infile} snips {start_snip} to {end_snip}")

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

    pos_df["relative_pos"] = np.arange(len(pos_df))
    # actually should not filter, since need for rows. instead save start and end snips for columns
    pos_df = pos_df[pos_df.BP.between(start_snip, end_snip)]

    if len(pos_df) == 0:
        print(f"No data found between snips {start_snip} and {end_snip}")
        return None

    lower_pos, upper_pos = pos_df.relative_pos[[0, -1]]
    sparse_mat = sparse_mat[lower_pos : upper_pos + 1, :]
    dense = sparse_mat.todense()
    group.require_dataset(
        "full",
        data=dense,
        compression="gzip",
        compression_opts=9,
        shape=dense.shape,
        dtype=dense.dtype,
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


def add_right_slice_to_df(df, right_slice):
    if df.columns[-1] not in right_slice.columns:
        return pd.concat((df, right_slice), axis=1)
    infill = right_slice.loc[:, : df.columns[-1]]
    extension = right_slice.loc[:, df.columns[-1] :].iloc[:, 1:]
    df.loc[
        infill.index[0] : infill.index[-1], infill.columns[0] : infill.columns[-1]
    ] = infill
    return pd.concat((df, extension), axis=1)


def add_bottom_slice_to_df(df, bottom_slice):
    if df.index[-1] not in bottom_slice.index:
        return pd.concat((df, bottom_slice), axis=0)
    infill = bottom_slice.loc[: df.index[-1], :]
    extension = bottom_slice.loc[df.index[-1] :, :].iloc[1:, :]
    df.loc[
        infill.index[0] : infill.index[-1], infill.columns[0] : infill.columns[-1]
    ] = infill
    return pd.concat((df, extension), axis=0)


def add_main_slice_to_df(df, main_slice):
    if df is None:
        return main_slice
    if main_slice.index[0] not in df.index:
        # TODO - THIS REQUIRES EXTRA HANDLING, WHICH MUST OCCUR OUTSIDE OF THIS FUNCTION
        # IN THIS CASE, THE BOTTOM RIGHT CORNER OF df HASN'T BEEN FILLED. WE NEED TO GO BACK AND FILL IT IN WITH OLD DATA
        # MAKE SURE BOTH INDEX/COLUMN START/END ARE IN RANGE, OTHERWISE WE NEED FANCY HANDLING (SPLIT AND CONCAT)

        # THIS IS HOPEFULLY HANDLED NOW
        return pd.concat((df, main_slice), axis=0)

    if main_slice.index[-1] in df.index:
        # everything is normal
        df.loc[
            main_slice.index[0] : main_slice.index[-1],
            main_slice.columns[0] : main_slice.columns[-1],
        ] = main_slice
    else:
        # main slice goes too far?
        df.loc[main_slice.index[0] :, main_slice.columns[0] :] = main_slice.loc[
            : df.index[-1], : df.index[-1]
        ]
        df = pd.concat(
            (df, main_slice.loc[df.index[-1] :, df.index[-1] :].iloc[1:, 1:]), axis=0
        )

    return df


def get_submatrix_from_chromosome_by_range_h5(
    chromosome_group, i_start, i_end, j_start, j_end
):
    intervals = []
    for subgroup in chromosome_group.values():
        if subgroup.name != "/aux":
            intervals.append((subgroup.attrs["start_snip"], subgroup.attrs["end_snip"]))

    intervals.sort(key=lambda x: x[0])

    df = None
    for interval in intervals:
        # INTERVALS ARE ONLY FOR J VALUES!!!
        i_overlap = find_overlap((i_start, i_end), interval)
        j_overlap = find_overlap((j_start, j_end), interval)

        group = chromosome_group[f"snip_{interval[0]}"]
        if i_overlap and j_overlap:
            full_overlap = (
                min(i_overlap[0], j_overlap[0]),
                max(i_overlap[1], j_overlap[1]),
            )

            # right slice - all i in overlap, all j > interval end
            if j_end > interval[1]:
                right_slice = get_horizontal_slice(
                    group, *i_overlap, max(j_start, interval[1]), j_end
                )
            else:
                right_slice = pd.DataFrame()

            # triangular slice - full triangle, make symmetric, then subselect - can probably be done more efficiently
            main_slice = get_horizontal_slice(group, *full_overlap, *full_overlap)
            main_slice = main_slice + main_slice.T
            np.fill_diagonal(main_slice.values, DIAGONAL_LD)
            main_slice = subselect(main_slice, *i_overlap, *j_overlap)

            # bottom slice - all j in overlap, all i > interval end

            if i_end > interval[1]:
                bottom_slice = get_horizontal_slice(
                    group, *j_overlap, max(i_start, interval[1]), i_end
                ).T
            else:
                bottom_slice = pd.DataFrame()

            # get right overlap, bottom overlap, triangle overlap

            df = add_main_slice_to_df(df, main_slice)

            df = add_right_slice_to_df(df, right_slice)
            df = add_bottom_slice_to_df(df, bottom_slice)
        elif i_overlap:
            bottom_slice = get_horizontal_slice(
                group, *i_overlap, j_start, min(j_end, interval[0])
            )
            df = pd.concat((df, bottom_slice), axis=1)
            pass
        elif j_overlap:
            # make empty shape
            right_slice = get_horizontal_slice(
                group, i_start, min(i_end, interval[0]), *j_overlap
            )
            df = pd.concat((df, right_slice), axis=0)

    return df.fillna(0)


def subselect(df, i_start, i_end, j_start, j_end):
    BP_list = df[[]].copy()
    BP_list["BP"] = df.index.str.split(".").str[1].astype(int)
    BP_list["relative_pos"] = np.arange(len(BP_list))
    row_start, row_end = BP_list[BP_list.BP.between(i_start, i_end)].relative_pos[
        [0, -1]
    ]
    col_start, col_end = BP_list[BP_list.BP.between(j_start, j_end)].relative_pos[
        [0, -1]
    ]
    return df.iloc[row_start : row_end + 1, col_start : col_end + 1]


def subselect_list(df, i_list, j_list):
    BP_list = df[[]].copy()
    BP_list["BP"] = df.index.str.split(".").str[1].astype(int)
    BP_list["relative_pos"] = np.arange(len(BP_list))
    row_inds = BP_list[BP_list.BP.isin(i_list)].relative_pos
    col_inds = BP_list[BP_list.BP.isin(j_list)].relative_pos
    return df.iloc[row_inds, col_inds]


def add_slice_to_matrix(matrix, slice):
    if matrix is None:
        return slice
    else:
        return pd.concat((matrix, slice), axis=1).fillna(0)


def get_vertical_slice(group, start_col, end_col, start_row, end_row):
    df_ld_snps = extract_metadata_df_from_group(group)

    col_positions = df_ld_snps[df_ld_snps.BP.between(start_col, end_col)].relative_pos
    # row positions is wrong, should use a universal list
    row_positions = df_ld_snps[df_ld_snps.BP.between(start_row, end_row)].relative_pos

    if len(col_positions) and len(row_positions):
        slice = group[FULL_MATRIX_NAME][
            row_positions[0] : row_positions[-1] + 1,
            col_positions[0] : col_positions[-1] + 1,
        ]
    else:
        slice = None

    return pd.DataFrame(
        slice,
        index=row_positions.index.astype(str),
        columns=col_positions.index.astype(str),
    )


def get_horizontal_slice(group, start_row, end_row, start_col, end_col):
    df_ld_snps = extract_metadata_df_from_group(group)

    row_positions = df_ld_snps[df_ld_snps.BP.between(start_row, end_row)].relative_pos
    col_positions = df_ld_snps[df_ld_snps.BP.between(start_col, end_col)].relative_pos

    if len(row_positions) and len(col_positions):
        slice = group[FULL_MATRIX_NAME][
            row_positions[0] : row_positions[-1] + 1,
            col_positions[0] : col_positions[-1] + 1,
        ]
    else:
        slice = None

    return pd.DataFrame(
        slice,
        index=row_positions.index.astype(str),
        columns=col_positions.index.astype(str),
    )


def get_horizontal_slice_by_list(group, row_list, col_list):
    df_ld_snps = extract_metadata_df_from_group(group)

    row_positions = df_ld_snps[df_ld_snps.BP.isin(row_list)].relative_pos
    col_positions = df_ld_snps[df_ld_snps.BP.isin(col_list)].relative_pos

    if len(row_positions) and len(col_positions):
        slice = group[FULL_MATRIX_NAME][row_positions.tolist()][:, col_positions]
    else:
        slice = None

    return pd.DataFrame(
        slice,
        index=row_positions.index.astype(str),
        columns=col_positions.index.astype(str),
    )


def list_overlap(index_list, interval):
    return [index for index in index_list if interval[0] <= index < interval[1]]


def get_submatrix_from_chromosome_by_list_h5(chromosome_group, i_list, j_list):
    if len(i_list) == 0 or len(j_list) == 0:
        return pd.DataFrame()
    i_list, j_list, ind_list = sort_and_combine_lists(i_list, j_list)

    intervals = []
    for subgroup in chromosome_group.values():
        if subgroup.name != "/aux":
            intervals.append((subgroup.attrs["start_snip"], subgroup.attrs["end_snip"]))

    intervals.sort(key=lambda x: x[0])

    df = None
    for interval in intervals:
        # INTERVALS ARE ONLY FOR J VALUES!!!
        i_overlap = list_overlap(i_list, interval)
        j_overlap = list_overlap(j_list, interval)

        group = chromosome_group[f"snip_{interval[0]}"]
        if i_overlap and j_overlap:
            full_overlap = list_overlap(ind_list, interval)

            # right slice - all i in overlap, all j > interval end
            if j_list[-1] > interval[1]:
                right_slice = get_horizontal_slice_by_list(
                    group, i_overlap, list_overlap(j_list, (interval[1], np.inf))
                )
            else:
                right_slice = pd.DataFrame()

            # triangular slice - full triangle, make symmetric, then subselect - can probably be done more efficiently
            main_slice = get_horizontal_slice_by_list(group, full_overlap, full_overlap)
            main_slice = main_slice + main_slice.T
            np.fill_diagonal(main_slice.values, DIAGONAL_LD)
            main_slice = subselect_list(main_slice, i_overlap, j_overlap)

            # bottom slice - all j in overlap, all i > interval end

            if i_list[-1] > interval[1]:
                bottom_slice = get_horizontal_slice_by_list(
                    group, j_overlap, list_overlap(i_list, (interval[1], np.inf))
                ).T
            else:
                bottom_slice = pd.DataFrame()

            # get right overlap, bottom overlap, triangle overlap
            df = add_main_slice_to_df(df, main_slice)

            df = add_right_slice_to_df(df, right_slice)
            df = add_bottom_slice_to_df(df, bottom_slice)
        elif i_overlap:
            bottom_slice = get_horizontal_slice_by_list(
                group, i_overlap, list_overlap(j_list, (0, interval[0]))
            )
            df = pd.concat((df, bottom_slice), axis=1)
            pass
        elif j_overlap:
            right_slice = get_horizontal_slice_by_list(
                group, list_overlap(i_list, (0, interval[0])), j_overlap
            )
            df = pd.concat((df, right_slice), axis=0)

    return df.fillna(0)


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
    print(f"Found {len(indices)} matching MAFs")
    maf_result = get_submatrix_from_chromosome_by_list_h5(
        chromosome_group, indices, indices
    )
    assert all(np.diagonal(maf_result) == 1)
    return maf_result


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
@click.option("--start_snip", "-s", type=int, default=None)
@click.option("--end_snip", "-e", type=int, default=None)
def convert(infile, outfile, precision, decimals, start_snip, end_snip):
    convert_h5(infile, outfile, precision, decimals, start_snip, end_snip)


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
    print(f"Converting chromosome {chromosome}")

    # f = h5py.File(ld_file, "r")

    filtered = []
    for file in os.listdir(directory):
        if (
            os.path.isfile(os.path.join(directory, file))
            and file.startswith(f"chr{chromosome}_")
            and file.endswith(".npz")
        ):
            filtered.append((file, int(file.split("_")[1])))

    filtered.sort(key=lambda x: x[1])

    start_snip = max(start_snip, filtered[0][1])

    first_missing_snip = start_snip

    for i, (file, snip) in enumerate(filtered):
        if snip >= start_snip:
            print(f"Converting {file}")
            if i + 1 < len(filtered):
                next_covered_snip = filtered[i + 1][1]
            else:
                next_covered_snip = np.inf
            convert_h5(
                os.path.join(directory, file),
                outfile,
                precision,
                decimals,
                first_missing_snip,
                next_covered_snip,
            )
            first_missing_snip = next_covered_snip


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
