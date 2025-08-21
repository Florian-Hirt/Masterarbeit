import subprocess, sys
def test_cli_help():
    out = subprocess.check_output([sys.executable, "-m", "masterarbeit.cli", "--help"]).decode()
    assert "preprocess" in out and "run" in out and "evaluate" in out
