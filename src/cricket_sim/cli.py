import argparse
import sys

from . import bronze, paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="cricket")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="fetch a source into a new bronze snapshot")
    p.add_argument("source", help="a source name from config, or all")
    p = sub.add_parser("lock", help="pin the snapshot builds read")
    p.add_argument("source")
    p.add_argument("snapshot_id")
    sub.add_parser("verify", help="checksum every locked snapshot")
    p = sub.add_parser("build", help="rebuild silver and gold from locked snapshots")
    p.add_argument("--allow-dirty", action="store_true", help="build from uncommitted code (recorded in run.json)")
    p.add_argument("--allow-shrink", action="store_true",
                   help="accept a table losing more than 1%% of its rows since the last build (recorded in run.json)")
    p = sub.add_parser("export", help="write the licence-checked dataset export to data/export/")
    p.add_argument("--allow-dirty", action="store_true", help="export a build made from uncommitted code (tests only)")
    p = sub.add_parser("model", help="walk-forward ODI win-probability and ball-outcome models -> data/models/")
    p.add_argument("--allow-dirty", action="store_true", help="run from uncommitted code (recorded in run.json)")
    p.add_argument("--first-season", type=int, default=2008, help="first season to predict (default 2008)")
    p.add_argument("--min-train-matches", type=int, default=300, help="skip seasons with less history (default 300)")
    p.add_argument("--holdout-from-season", type=int, default=2023,
                   help="locked holdout: one fit on clean seasons before this, scored once from it on (default 2023)")
    sub.add_parser("contracts", help="lint every data contract and check the built tables against them")
    sub.add_parser("docs", help="regenerate the data dictionary and lineage graph from the contracts")
    sub.add_parser("backup", help="copy bronze to the rclone remote")
    sub.add_parser("restore", help="fetch locked bronze snapshots from the rclone remote")
    sub.add_parser("status", help="show locked snapshots and local availability")
    p = sub.add_parser("prune", help="free disk: drop retained previous generations, failed stages and quarantines")
    p.add_argument("--keep", type=int, default=1, help="previous generations to keep per kind (default 1; 0 keeps none)")
    p.add_argument("--dry-run", action="store_true", help="list what would be removed without removing it")
    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        from .ingest import INGESTERS

        names = list(INGESTERS) if args.source == "all" else [args.source]
        for name in names:
            INGESTERS[name]()
    elif args.cmd == "lock":
        bronze.lock_source(args.source, args.snapshot_id)
        print(f"locked {args.source} -> {args.snapshot_id}")
    elif args.cmd == "verify":
        problems = bronze.verify_all()
        print("\n".join(problems) or "all locked snapshots verify")
        sys.exit(1 if problems else 0)
    elif args.cmd == "build":
        from .build import build

        build(allow_dirty=args.allow_dirty, allow_shrink=args.allow_shrink)
    elif args.cmd == "export":
        from .export import export

        export(allow_dirty=args.allow_dirty)
    elif args.cmd == "model":
        from .models.run import run

        run(first_season=args.first_season, min_train_matches=args.min_train_matches, allow_dirty=args.allow_dirty,
            holdout_from_season=args.holdout_from_season)
    elif args.cmd == "contracts":
        import duckdb

        from . import contracts

        contracts.refuse_half_published()
        problems = contracts.lint()
        if problems:
            print("\n".join(problems))
            sys.exit(1)
        con = duckdb.connect()
        print(contracts.enforce(con))
        # a models/ folder, even an empty one, is checked; only a layer that was never built is skipped, out loud
        if paths.MODELS.exists() or any(c.layer == "models" and c.data_path.exists() for c in contracts.load_all().values()):
            print(contracts.enforce(con, layers=("models",)))
        else:
            print("models layer: not built, not checked (run `cricket model`)")
    elif args.cmd == "docs":
        from .docs import write_docs

        write_docs()
    elif args.cmd in ("backup", "restore"):
        from . import remote

        getattr(remote, args.cmd)()
    elif args.cmd == "prune":
        from .transaction import prune

        freed = prune(paths.DATA, keep=args.keep, dry_run=args.dry_run)
        for path, size in freed.items():
            print(f"{'would remove' if args.dry_run else 'removed'} {path} ({size / 1e6:.0f} MB)")
        print(f"{'would free' if args.dry_run else 'freed'} {sum(freed.values()) / 1e9:.2f} GB in {len(freed)} item(s)")
    elif args.cmd == "status":
        for source, sid in sorted(bronze.read_lock().items()):
            local = "ok" if not bronze.verify(source, sid) else "MISSING/CORRUPT"
            others = [s for s in bronze.list_snapshots(source) if s != sid]
            print(f"{source:10} locked={sid:12} local={local}  other snapshots={others}")


if __name__ == "__main__":
    main()
