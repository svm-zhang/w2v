import argparse


def parse_cmd() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    cmd = parser.add_subparsers(title="Command", dest="command")
    train = cmd.add_parser(name="train")
    train.add_argument(
        "--input",
        metavar="FILE",
        required=True,
        help="Specify training text file.",
    )
    train.add_argument(
        "--outp",
        metavar="STR",
        required=True,
        help="Specify output prefix.",
    )
    train.add_argument(
        "--model",
        metavar="STR",
        default="ngram",
        choices=["ngram", "skip_gram"],
        help="Specify model to train.",
    )

    return parser
