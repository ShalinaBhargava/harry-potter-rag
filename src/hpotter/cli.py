import argparse


def main():
    parser = argparse.ArgumentParser(prog="hpotter")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="rebuild the Chroma index")
    sub.add_parser("query", help="interactive question answering")
    sub.add_parser("eval", help="run the retrieval eval")

    args = parser.parse_args()
    if args.command == "ingest":
        from hpotter.ingest import main as run
    elif args.command == "query":
        from hpotter.query import main as run
    elif args.command == "eval":
        from hpotter.run_evals import main as run
    run()


if __name__ == "__main__":
    main()