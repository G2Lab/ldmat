import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import os
import pandas as pd
import click
import shutil
from heapq import merge
import h5py

DIAGONAL_LD = 1
DEFAULT_BLOCK_SIZE = 2000
MIN_BLOCK_SIZE = 100
UPPER_BOUND_HEURISTIC_STEP = 100

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


def sort_and_combine_lists(a, b):
    sorted_a, sorted_b = sorted(set(a)), sorted(set(b))
    return sorted_a, sorted_b, list(merge(sorted_a, sorted_b))


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


def heuristic_block_size(matrix, precision, threshold=0.5):
    if precision == 0:
        step = UPPER_BOUND_HEURISTIC_STEP
    else:
        step = min(UPPER_BOUND_HEURISTIC_STEP, int(1 / precision))
    for i in range(step, matrix.shape[0], step):
        print(f"Checking diagonal {i}")
        if (np.count_nonzero(matrix.diagonal(i)) / i) < threshold:
            return max(i, MIN_BLOCK_SIZE)
    return DEFAULT_BLOCK_SIZE


def log_block_size(matrix, precision):
    if precision == 0:
        step = UPPER_BOUND_HEURISTIC_STEP
    else:
        step = min(UPPER_BOUND_HEURISTIC_STEP, int(1 / precision))

    lg = np.log(matrix.shape[0])
    for i in range(step, matrix.shape[0], step):
        print(f"Checking diagonal {i}")
        if np.count_nonzero(matrix.diagonal(i)) < lg:
            return max(i, MIN_BLOCK_SIZE)
    return DEFAULT_BLOCK_SIZE


def convert(infile, outdir, block_size=None, precision=0, heuristic=None, decimals=3):
    base_infile = os.path.splitext(infile)[0]
    filename = base_infile.split("/")[-1]
    chromosome, start_snip, end_snip = filename.split("_")
    start_snip, end_snip = int(start_snip), int(end_snip)
    chromosome_dir = os.path.join(outdir, chromosome + f"_{precision}")
    if not os.path.exists(chromosome_dir):
        os.makedirs(chromosome_dir)
    outdir = os.path.join(chromosome_dir, f"{start_snip}_{end_snip}")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sparse_mat = sparse.triu(sparse.load_npz(base_infile + ".npz").T, format="csr")
    sparse_mat.setdiag(0)
    sparse_mat.data = np.round(sparse_mat.data, decimals)
    sparse_mat = adjust_to_zero(sparse_mat, precision)
    mat_size = sparse_mat.shape[0]

    if not block_size:
        print("Calculating block size")
        if heuristic == "log":
            block_size = log_block_size(sparse_mat, precision)
        else:
            block_size = heuristic_block_size(sparse_mat, precision)

    shutil.copy(base_infile + ".gz", os.path.join(outdir, "metadata.gz"))

    pd.DataFrame(
        [[start_snip, end_snip, block_size, precision]],
        columns=["start_snip", "end_snip", "block_size", "precision"],
    ).to_csv(os.path.join(outdir, "META"), index=False)

    for i in range(0, mat_size, block_size):
        if i + block_size < mat_size:
            off_diagonal = sparse_mat[i : i + block_size, i + block_size :]
            # or is tocoo() better?
            print(f"Writing off diagonal {i}")
            sparse.save_npz(
                os.path.join(outdir, f"row_{i}"), off_diagonal, compressed=True
            )
            print("Finished writing")

        reduced = sparse_mat[i : i + block_size, i : i + block_size]
        print(f"Writing block {i}")
        sparse.save_npz(os.path.join(outdir, f"block_{i}"), reduced, compressed=True)
        print("Finished writing")


def convert_h5(
    infile, outfile, block_size=None, precision=0, heuristic=None, decimals=3
):
    base_infile = os.path.splitext(infile)[0]
    filename = base_infile.split("/")[-1]
    chromosome, start_snip, end_snip = filename.split("_")
    start_snip, end_snip = int(start_snip), int(end_snip)

    f = h5py.File(outfile, "a")

    group = f.require_group(f"{chromosome}/snips_{start_snip}_{end_snip}")

    sparse_mat = sparse.triu(sparse.load_npz(base_infile + ".npz").T, format="csr")
    sparse_mat.setdiag(0)
    sparse_mat.data = np.round(sparse_mat.data, decimals)
    sparse_mat = adjust_to_zero(sparse_mat, precision)
    mat_size = sparse_mat.shape[0]

    if not block_size:
        print("Calculating block size")
        if heuristic == "log":
            block_size = log_block_size(sparse_mat, precision)
        else:
            block_size = heuristic_block_size(sparse_mat, precision)

    pos_df = metadata_to_df(base_infile + ".gz")
    group.require_dataset(
        "positions",
        data=pos_df,
        compression="gzip",
        shape=pos_df.shape,
        dtype=pos_df.dtypes[0],
    )
    names = pos_df.index.to_numpy().astype("S")
    group.create_dataset("names", data=names, compression="gzip")

    group.attrs["start_snip"] = start_snip
    group.attrs["end_snip"] = end_snip
    group.attrs["block_size"] = block_size
    group.attrs["precision"] = precision

    for i in range(0, mat_size, block_size):
        if i + block_size < mat_size:
            off_diagonal = sparse_mat[i : i + block_size, i + block_size :].todense()
            # or is tocoo() better?
            print(f"Writing off diagonal {i}")
            group.require_dataset(
                f"row_{i}",
                data=off_diagonal,
                compression="gzip",
                compression_opts=9,
                shape=off_diagonal.shape,
                dtype=off_diagonal.dtype,
                scaleoffset=decimals,
            )
            print("Finished writing")

        reduced = sparse_mat[i : i + block_size, i : i + block_size].todense()
        print(f"Writing block {i}")
        group.require_dataset(
            f"block_{i}",
            data=reduced,
            compression="gzip",
            compression_opts=9,
            shape=reduced.shape,
            dtype=reduced.dtype,
            scaleoffset=decimals,
        )
        print("Finished writing")


def convert_h5_no_block(
    infile, outfile, block_size=None, precision=0, heuristic=None, decimals=3
):
    base_infile = os.path.splitext(infile)[0]
    filename = base_infile.split("/")[-1]
    chromosome, start_snip, end_snip = filename.split("_")
    start_snip, end_snip = int(start_snip), int(end_snip)

    f = h5py.File(outfile, "a")

    group = f.require_group(f"{chromosome}/snips_{start_snip}_{end_snip}")

    sparse_mat = sparse.triu(sparse.load_npz(base_infile + ".npz").T, format="csr")
    sparse_mat.setdiag(0)
    sparse_mat.data = np.round(sparse_mat.data, decimals)
    sparse_mat = adjust_to_zero(sparse_mat, precision)
    mat_size = sparse_mat.shape[0]

    pos_df = metadata_to_df(base_infile + ".gz")
    group.require_dataset(
        "positions",
        data=pos_df,
        compression="gzip",
        shape=pos_df.shape,
        dtype=pos_df.dtypes[0],
    )
    names = pos_df.index.to_numpy().astype("S")
    group.create_dataset("names", data=names, compression="gzip")

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


def get_metadata(dir):
    metadata = pd.read_csv(
        dir + "/META",
        dtype={
            "start_snip": int,
            "end_snip": int,
            "block_size": int,
            "precision": float,
        },
    )
    return (
        metadata.start_snip[0],
        metadata.end_snip[0],
        metadata.block_size[0],
        metadata.precision[0],
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


def load_snips_df(dir):
    # load the SNPs metadata
    gz_file = dir + "/metadata.gz"
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
    df_ld_snps.rename_axis("relative_pos", inplace=True)
    df_ld_snps.reset_index(inplace=True)
    df_ld_snps.index = (
        df_ld_snps["CHR"].astype(str)
        + "."
        + df_ld_snps["BP"].astype(str)
        + "."
        + df_ld_snps["A1"]
        + "."
        + df_ld_snps["A2"]
    )
    return df_ld_snps


def find_interval_index(val, intervals, start_index):
    index = start_index
    start_snip, end_snip = intervals[start_index]
    while val >= end_snip:
        index += 1
        if index == len(intervals):
            return None
        start_snip, end_snip = intervals[index]
    return index


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
                    chromosome_group[f"snips_{interval[0]}_{interval[1]}"],
                    *i_overlap,
                    *j_overlap,
                )
            )
    return combine_matrices(submatrices)


def get_submatrix_from_chromosome_by_range(
    chromosome_dir, i_start, i_end, j_start, j_end
):
    intervals = []
    for subdir in [x[0] for x in os.walk(chromosome_dir)][1:]:
        dirname = subdir.split("/")[-1]
        start, end = dirname.split("_")
        intervals.append((int(start), int(end)))

    intervals.sort(key=lambda x: x[0])

    submatrices = []
    for interval in intervals:
        i_overlap = find_overlap((i_start, i_end), interval)
        j_overlap = find_overlap((j_start, j_end), interval)
        if i_overlap or j_overlap:
            i_overlap = i_overlap or (-1, -1)
            j_overlap = j_overlap or (-1, -1)
            submatrices.append(
                get_submatrix_by_ranges(
                    os.path.join(chromosome_dir, f"{interval[0]}_{interval[1]}"),
                    *i_overlap,
                    *j_overlap,
                )
            )
    return combine_matrices(submatrices)


def get_submatrix_from_chromosome_by_list(chromosome_dir, i_list, j_list):
    if len(i_list) == 0 or len(j_list) == 0:
        return pd.DataFrame()
    i_list, j_list, ind_list = sort_and_combine_lists(i_list, j_list)

    # need to find all intervals and compare

    intervals = []
    for subdir in [x[0] for x in os.walk(chromosome_dir)][1:]:
        dirname = subdir.split("/")[-1]
        start, end = dirname.split("_")
        intervals.append((int(start), int(end)))

    intervals.sort(key=lambda x: x[0])

    submatrices = []
    prev = 0
    interval_ind = find_interval_index(ind_list[0], intervals, 0)
    start_snip, end_snip = intervals[interval_ind]
    for i, ind in enumerate(ind_list):
        if ind >= end_snip:
            submatrices.append(
                get_submatrix_by_indices(
                    os.path.join(chromosome_dir, f"{start_snip}_{end_snip}"),
                    ind_list[prev:i],
                    ind_list[prev:i],
                )
            )
            prev = i
            interval_ind = find_interval_index(ind, intervals, interval_ind)
            if interval_ind is None:
                # can end early
                return combine_matrices(submatrices)
            start_snip, end_snip = intervals[interval_ind]

    submatrices.append(
        get_submatrix_by_indices(
            os.path.join(chromosome_dir, f"{start_snip}_{end_snip}"),
            ind_list[prev:],
            ind_list[prev:],
        )
    )
    return combine_matrices(submatrices)


def load_symmetric_matrix(dir, index_df):
    start_snip, end_snip, block_size, precision = get_metadata(dir)

    ind_blocks = index_df.floordiv(block_size) * block_size

    rows = []

    for block in range(0, end_snip - start_snip, block_size):
        early_inds = index_df[ind_blocks < block]
        local_inds = index_df[ind_blocks == block]
        late_inds = index_df[ind_blocks > block]

        if len(local_inds) == 0:
            continue

        local_ind_offsets = list(local_inds.mod(block_size))

        row = [np.zeros((len(local_inds), len(early_inds)))]

        block_matrix = sparse.load_npz(
            f"{dir}/block_{block}.npz"
        )  # could be avoided when selecting non overlapping
        row.append(block_matrix[np.ix_(local_ind_offsets, local_ind_offsets)].todense())

        if len(late_inds):
            aux_matrix = sparse.load_npz(f"{dir}/row_{block}.npz")
            row.append(
                aux_matrix[
                    np.ix_(local_ind_offsets, late_inds - (block + block_size))
                ].todense()
            )

        rows.append(np.hstack(row))

    if not len(rows):
        return np.empty((0, 0))
    triangular = np.vstack(rows)
    symmetric = triangular + triangular.T
    np.fill_diagonal(symmetric, DIAGONAL_LD)
    return symmetric


def load_symmetric_matrix_h5(group, index_df):
    start_snip = group.attrs["start_snip"]
    end_snip = group.attrs["end_snip"]
    block_size = group.attrs["block_size"]

    ind_blocks = index_df.floordiv(block_size) * block_size

    rows = []

    for block in range(0, end_snip - start_snip, block_size):
        early_inds = index_df[ind_blocks < block]
        local_inds = index_df[ind_blocks == block]
        late_inds = index_df[ind_blocks > block]

        if len(local_inds) == 0:
            continue

        local_ind_offsets = list(local_inds.mod(block_size))

        row = [np.zeros((len(local_inds), len(early_inds)))]

        block_matrix = group[f"block_{block}"]
        print(f"Reading from block {block}")
        block_matrix = block_matrix[:, local_ind_offsets][local_ind_offsets]
        row.append(block_matrix)

        if len(late_inds):
            aux_matrix = group[f"row_{block}"]
            print(f"Reading from offset {block}")
            row.append(
                aux_matrix[:, (late_inds - (block + block_size)).tolist()][
                    local_ind_offsets
                ]
            )

        rows.append(np.hstack(row))

    if not len(rows):
        return np.empty((0, 0))
    triangular = np.vstack(rows)
    symmetric = triangular + triangular.T
    np.fill_diagonal(symmetric, DIAGONAL_LD)
    return symmetric


def construct_labeled_df(full_matrix, i_index_df, j_index_df):
    # submatrix = full_matrix[np.ix_(j_index_df, i_index_df)]
    submatrix = full_matrix[: len(j_index_df), : len(i_index_df)]
    return pd.DataFrame(submatrix, index=j_index_df.index, columns=i_index_df.index)


def get_submatrix_by_indices(dir, i_list, j_list):
    i_list, j_list, ind_list = sort_and_combine_lists(i_list, j_list)

    df_ld_snps = load_snips_df(dir)

    ind_temp = df_ld_snps[df_ld_snps.BP.isin(ind_list)].relative_pos

    i_temp = df_ld_snps[df_ld_snps.BP.isin(i_list)].relative_pos
    j_temp = df_ld_snps[df_ld_snps.BP.isin(j_list)].relative_pos

    return construct_labeled_df(load_symmetric_matrix(dir, ind_temp), i_temp, j_temp)


def get_submatrix_by_ranges_h5(group, i_start, i_end, j_start, j_end):
    df_ld_snps = extract_metadata_df_from_group(group)

    ind_temp = df_ld_snps[
        df_ld_snps.BP.between(i_start, i_end) | df_ld_snps.BP.between(j_start, j_end)
    ].relative_pos

    i_temp = df_ld_snps[df_ld_snps.BP.between(i_start, i_end)].relative_pos
    j_temp = df_ld_snps[df_ld_snps.BP.between(j_start, j_end)].relative_pos

    return construct_labeled_df(
        load_symmetric_matrix_h5(group, ind_temp), i_temp, j_temp
    )


def get_submatrix_by_ranges(dir, i_start, i_end, j_start, j_end):
    df_ld_snps = load_snips_df(dir)

    ind_temp = df_ld_snps[
        df_ld_snps.BP.between(i_start, i_end) | df_ld_snps.BP.between(j_start, j_end)
    ].relative_pos

    i_temp = df_ld_snps[df_ld_snps.BP.between(i_start, i_end)].relative_pos
    j_temp = df_ld_snps[df_ld_snps.BP.between(j_start, j_end)].relative_pos

    return construct_labeled_df(load_symmetric_matrix(dir, ind_temp), i_temp, j_temp)


def get_value_at_index(dir, i, j):
    return get_submatrix_by_indices(dir, [i], [j])


def convert_maf(infile, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    base_infile = os.path.splitext(infile)[0]
    filename = base_infile.split("/")[-1]
    chromosome = filename.split("_")[-2]
    outfile = os.path.join(outdir, chromosome + ".csv")

    maf = pd.read_csv(infile, sep="\t", header=None, names=MAF_COLS)
    maf = maf.sort_values("MAF")
    maf[["Position", "MAF"]].to_csv(outfile, index=False)


def get_maf_indices_by_range(maf_file, lower_bound, upper_bound):
    # inclusive

    maf = pd.read_csv(maf_file, index_col="Position")
    start_index = maf.MAF.searchsorted(lower_bound)
    while start_index > 0 and maf.MAF.iloc[start_index - 1] >= lower_bound:
        start_index -= 1

    end_index = maf.MAF.searchsorted(upper_bound)
    while end_index < len(maf) and maf.MAF.iloc[end_index + 1] <= upper_bound:
        end_index += 1

    return list(maf.index[start_index:end_index])


def get_submatrix_by_maf_range(dir, maf_file, lower_bound, upper_bound):
    indices = get_maf_indices_by_range(maf_file, lower_bound, upper_bound)
    return get_submatrix_from_chromosome_by_list(dir, indices, indices)


# maybe make an object


@click.group()
def cli():
    pass


@cli.command()
@click.argument("chromosome_dir")
@click.option("--i_start", type=int)
@click.option("--i_end", type=int)
@click.option("--j_start", type=int)
@click.option("--j_end", type=int)
@click.option("--outfile", "-o", default=None)
@click.option("--symmetric", "-s", is_flag=True, default=False)
def submatrix(chromosome_dir, i_start, i_end, j_start, j_end, outfile, symmetric):
    if symmetric and (j_start is not None or j_end is not None):
        raise ValueError("Symmetric flag only compatible with i indexing.")
    if symmetric:
        j_start, j_end = i_start, i_end
    res = get_submatrix_from_chromosome_by_range(
        chromosome_dir, i_start, i_end, j_start, j_end
    )
    if outfile:
        # name index?
        res.to_csv(outfile)
    else:
        print(res)


@cli.command()
@click.argument("ld_file")
@click.argument("chromosome")
@click.option("--i_start", type=int)
@click.option("--i_end", type=int)
@click.option("--j_start", type=int)
@click.option("--j_end", type=int)
@click.option("--outfile", "-o", default=None)
@click.option("--symmetric", "-s", is_flag=True, default=False)
def submatrix_h5(
    ld_file, chromosome, i_start, i_end, j_start, j_end, outfile, symmetric
):
    if symmetric and (j_start is not None or j_end is not None):
        raise ValueError("Symmetric flag only compatible with i indexing.")
    if symmetric:
        j_start, j_end = i_start, i_end
    ld = h5py.File(ld_file, "r")
    res = get_submatrix_from_chromosome_by_range_h5(
        ld[f"chr{chromosome}"], i_start, i_end, j_start, j_end
    )
    if outfile:
        # name index?
        res.to_csv(outfile)
    else:
        print(res)


@cli.command()
@click.argument("infile")
@click.argument("outdir")
@click.option("--block_size", "-b", type=int, default=None)
@click.option("--precision", "-p", type=float, default=0)
@click.option("--heuristic", "-h", type=str, default=None)
def convert_file(infile, outdir, block_size, precision, heuristic):
    convert(infile, outdir, block_size, precision, heuristic)


@cli.command()
@click.argument("infile")
@click.argument("outfile")
@click.option("--block_size", "-b", type=int, default=None)
@click.option("--precision", "-p", type=float, default=0)
@click.option("--heuristic", "-h", type=str, default=None)
@click.option("--decimals", "-d", type=int, default=3)
def convert_file_h5(infile, outfile, block_size, precision, heuristic, decimals):
    convert_h5(infile, outfile, block_size, precision, heuristic, decimals)


@cli.command()
@click.argument("infile")
@click.argument("outfile")
@click.option("--block_size", "-b", type=int, default=None)
@click.option("--precision", "-p", type=float, default=0)
@click.option("--heuristic", "-h", type=str, default=None)
@click.option("--decimals", "-d", type=int, default=3)
def convert_file_h5_simple(infile, outfile, block_size, precision, heuristic, decimals):
    convert_h5_no_block(infile, outfile, block_size, precision, heuristic, decimals)


@cli.command()
@click.argument("chromosome_dir")
@click.argument("maf_file")
@click.option("--lower_bound", "-l", type=float, default=0)
@click.option("--upper_bound", "-u", type=float, default=0.5)
@click.option("--outfile", "-o", default=None)
def submatrix_by_maf(chromosome_dir, maf_file, lower_bound, upper_bound, outfile):
    res = get_submatrix_by_maf_range(chromosome_dir, maf_file, lower_bound, upper_bound)
    if outfile:
        # name index?
        res.to_csv(outfile)
    else:
        print(res)


if __name__ == "__main__":
    cli()

# example: python3 main.py data/processed/chr1_0 --i_start 10000 --i_end 100000 -s -o /tmp/test.csv
