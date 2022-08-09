import filecmp

from click.testing import CliRunner

from src.ldmat.__main__ import submatrix, submatrix_by_maf


def test_npz_query():
    tmp_output = "/tmp/out.npz"

    runner = CliRunner()
    result = runner.invoke(
        submatrix,
        [
            "examples/chr21_partial.h5",
            "--i-start",
            14900001,
            "--i-end",
            15100001,
            "-o",
            tmp_output,
        ],
    )
    assert result.exit_code == 0, print(result.output)

    assert filecmp.cmp(tmp_output, "examples/query_result.npz")


def test_csv_query():
    tmp_output = "/tmp/out.csv"

    runner = CliRunner()
    result = runner.invoke(
        submatrix,
        [
            "examples/chr21_partial.h5",
            "--i-start",
            14900001,
            "--i-end",
            15100001,
            "-o",
            tmp_output,
        ],
    )
    assert result.exit_code == 0, print(result.output)

    assert filecmp.cmp(tmp_output, "examples/query_result.csv")


def test_maf_query():
    tmp_output = "/tmp/maf_out.npz"

    runner = CliRunner()
    result = runner.invoke(
        submatrix_by_maf,
        ["examples/chr21_partial.h5", "-l", 0.2, "-u", 0.22, "-o", tmp_output],
    )
    assert result.exit_code == 0, print(result.output)

    assert filecmp.cmp(tmp_output, "examples/query_result_maf.npz")
