from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .run_im import build_parser as build_im_parser
from .run_im import run_im
from .run_all import build_parser as build_all_parser
from .run_all import run_all
from .run_mc import build_parser as build_mc_parser
from .run_mc import run_mc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EPOL modified experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="problem", required=True)

    im_parser = subparsers.add_parser(
        "im",
        help="Run Influence Maximization problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_im_parser(im_parser)
    im_parser.set_defaults(handler=run_im)

    mc_parser = subparsers.add_parser(
        "mc",
        help="Run Maximum Coverage problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_mc_parser(mc_parser)
    mc_parser.set_defaults(handler=run_mc)

    all_parser = subparsers.add_parser(
        "all",
        help="Run grid-search style experiments across IM/MC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_all_parser(all_parser)
    all_parser.set_defaults(handler=run_all)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and len(sys.argv) == 1:
        argv = ["im"]
    elif argv is not None and len(argv) == 0:
        argv = ["im"]

    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
