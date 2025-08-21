import argparse

def cmd_preprocess(args):
    print("TODO: hook this to your preprocessing (e.g., SAFIM or code graph steps).")

def cmd_run(args):
    # If a nested run subcommand was chosen, it'll be in args.run_cmd
    if getattr(args, "run_cmd", None) == "sample":
        print("Running sample pipeline… (placeholder)")
    else:
        print("Running main experiment… (placeholder)")

def cmd_evaluate(args):
    print("TODO: hook this to your evaluation/plotting pipeline.")

def build_parser():
    p = argparse.ArgumentParser(prog="masterarbeit", description="Paper code CLI (skeleton)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # preprocess
    sp = sub.add_parser("preprocess", help="Prepare data / intermediate artifacts")
    sp.set_defaults(func=cmd_preprocess)

    # run (with optional nested subcommands)
    sr = sub.add_parser("run", help="Run an experiment")
    sr.set_defaults(func=cmd_run)
    sr_sub = sr.add_subparsers(dest="run_cmd")
    sr_sub.required = False  # allow `masterarbeit run` without picking a nested subcommand

    # example nested subcommand
    srs = sr_sub.add_parser("sample", help="Example subcommand placeholder")
    srs.set_defaults(run_cmd="sample")

    # evaluate
    se = sub.add_parser("evaluate", help="Evaluate results / make plots")
    se.set_defaults(func=cmd_evaluate)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
