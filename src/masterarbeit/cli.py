import argparse
from . import preprocess as _pre
from . import runner as _run
from . import eval as _eval

def cmd_preprocess(args):
    _pre.main(args)

def cmd_run(args):
    # If a nested run subcommand was chosen, it'll be in args.run_cmd
    if getattr(args, "run_cmd", None) == "sample":
        _run.main(args, mode="sample")
    else:
        _run.main(args, mode=None)

def cmd_evaluate(args):
    _eval.main(args)

def build_parser():
    p = argparse.ArgumentParser(prog="masterarbeit", description="Paper code CLI (skeleton)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preprocess", help="Prepare data / intermediate artifacts")
    sp.set_defaults(func=cmd_preprocess)

    sr = sub.add_parser("run", help="Run an experiment")
    sr.set_defaults(func=cmd_run)
    sr_sub = sr.add_subparsers(dest="run_cmd")
    sr_sub.add_parser("sample", help="Example subcommand placeholder")

    se = sub.add_parser("evaluate", help="Evaluate results / make plots")
    se.set_defaults(func=cmd_evaluate)
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    # Safety: if no subcommand was chosen, print help.
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
