from click.testing import CliRunner

from lazychemvis.cli.create_cli import cli


def test_group_lists_both_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "fit" in result.output and "transform" in result.output


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "lazychemvis" in result.output


def test_transform_requires_input_reference_and_output():
    result = CliRunner().invoke(cli, ["transform", "-i", "x.csv", "-r", "space"])
    assert result.exit_code != 0
    assert "--output" in result.output


def test_fit_requires_tmap_env():
    result = CliRunner().invoke(cli, ["fit", "-i", "x.csv", "-r", "space"])
    assert result.exit_code != 0
    assert "--tmap-env" in result.output


def test_old_snake_case_flags_are_gone():
    result = CliRunner().invoke(cli, ["transform", "--lib_input", "x.csv"])
    assert result.exit_code != 0
    assert "No such option" in result.output
