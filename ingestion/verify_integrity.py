"""
verify_integrity.py - Cross-check Neo4j (graph) against Qdrant (vectors),
and optionally both against a fresh parse of the source repo.

Run from the ingestion/ folder, after run_pipeline.py has finished:

    python verify_integrity.py
    python verify_integrity.py --skip-vectors      # faster, skips vector sanity checks
    python verify_integrity.py --skip-source       # don't re-parse the repo
    python verify_integrity.py --max-show 25       # show more sample rows per issue

Exit code is 0 if every check passes, 1 if any check fails (warnings don't fail).
"""
import argparse
import hashlib
import math
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Fields that exist in BOTH stores and must agree, per node kind.
CLASS_FIELDS = ["name", "file_path", "signature", "docstring",
                "start_line", "end_line", "inherits_from", "code_content"]
FUNC_FIELDS = ["name", "type", "file_path", "belongs_to_class", "signature",
               "docstring", "is_api_endpoint", "api_path", "http_method",
               "start_line", "end_line", "code_content"]


# ----------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------
class Report:
    def __init__(self, max_show):
        self.max_show = max_show
        self.failures = 0
        self.warnings = 0

    def section(self, title):
        print(f"\n--- {title} " + "-" * max(3, 66 - len(title)))

    def info(self, msg):
        print(f"  [INFO] {msg}")

    def ok(self, msg):
        print(f"  [OK]   {msg}")

    def warn(self, msg, items=None):
        self.warnings += 1
        print(f"  [WARN] {msg}")
        self._show(items)

    def fail(self, msg, items=None):
        self.failures += 1
        print(f"  [FAIL] {msg}")
        self._show(items)

    def check(self, passed, ok_msg, fail_msg, items=None):
        if passed:
            self.ok(ok_msg)
        else:
            self.fail(fail_msg, items)

    def _show(self, items):
        if not items:
            return
        items = list(items)
        for it in items[: self.max_show]:
            print(f"           - {it}")
        if len(items) > self.max_show:
            print(f"           ... and {len(items) - self.max_show} more")


def norm(v):
    """Treat None, '' and [] as the same 'empty' value."""
    if v is None or v == "" or v == []:
        return None
    if isinstance(v, tuple):
        return list(v)
    return v


def digest(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


def short(v, n=60):
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_neo4j(args):
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        with driver.session(database=args.neo4j_database) as s:
            def q(cypher):
                return s.run(cypher).data()

            data = {
                "files": [r["p"] for r in q("MATCH (n:File) RETURN n {.*} AS p")],
                "classes": [r["p"] for r in q("MATCH (n:Class) RETURN n {.*} AS p")],
                "functions": [r["p"] for r in q("MATCH (n:Function) RETURN n {.*} AS p")],
                "contains": q(
                    "MATCH (a)-[:CONTAINS]->(b) "
                    "RETURN labels(a)[0] AS la, a.id AS ia, labels(b)[0] AS lb, b.id AS ib"),
                "inherits": q(
                    "MATCH (a:Class)-[:INHERITS_FROM]->(b:Class) RETURN a.id AS a, b.id AS b"),
                "calls": q(
                    "MATCH (a:Function)-[:CALLS]->(b:Function) RETURN a.id AS a, b.id AS b"),
                "orphans": [r["id"] for r in q(
                    "MATCH (n) WHERE (n:Class OR n:Function) "
                    "AND NOT EXISTS { ()-[:CONTAINS]->(n) } RETURN n.id AS id")],
                "no_file": [r["id"] for r in q(
                    "MATCH (n) WHERE (n:Class OR n:Function) "
                    "AND NOT EXISTS { MATCH (f:File {file_path: n.file_path}) } RETURN n.id AS id")],
                "empty_files": [r["id"] for r in q(
                    "MATCH (f:File) WHERE NOT EXISTS { (f)-[:CONTAINS]->() } RETURN f.id AS id")],
            }
    finally:
        driver.close()
    return data


def load_qdrant(args, with_vectors):
    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key, timeout=120)
    info = client.get_collection(args.qdrant_collection)
    exact_count = client.count(args.qdrant_collection, exact=True).count

    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=args.qdrant_collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        points.extend(batch)
        if offset is None:
            break

    dim = None
    try:
        vp = info.config.params.vectors
        dim = getattr(vp, "size", None)
    except Exception:
        pass
    client.close()
    return {"points": points, "count": exact_count, "dim": dim or 1024}


def load_source(repo_path):
    from ingestion.parser import process_repository
    parsed = process_repository(str(repo_path))
    by_fqn = {}
    counts = Counter()
    for c in parsed["chunks"]:
        counts[c["fqn"]] += 1
        by_fqn[c["fqn"]] = c  # last one wins, same as MERGE / upsert
    return parsed, by_fqn, counts


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_counts_and_sets(rep, neo_nodes, qd_by_fqn, qd_count, n_points, dup_qd):
    rep.section("1. Node / point counts and ID sets")
    n_cls = sum(1 for n in neo_nodes.values() if n["_kind"] == "Class")
    n_fn = sum(1 for n in neo_nodes.values() if n["_kind"] == "Function")
    rep.info(f"Neo4j: {n_cls} Class + {n_fn} Function = {len(neo_nodes)} chunk nodes")
    rep.info(f"Qdrant: {qd_count} points ({len(qd_by_fqn)} unique fqn payloads)")

    rep.check(len(neo_nodes) == qd_count,
              "Neo4j chunk-node count equals Qdrant point count",
              f"Count mismatch: Neo4j={len(neo_nodes)} vs Qdrant={qd_count}")
    rep.check(n_points == qd_count,
              "Scrolled every Qdrant point (scroll total == exact count)",
              f"Scroll returned {n_points} points but count says {qd_count}")
    rep.check(not dup_qd, "No duplicate fqn across Qdrant points",
              "Duplicate fqn values across different Qdrant point IDs", dup_qd)

    neo_ids, qd_ids = set(neo_nodes), set(qd_by_fqn)
    missing_in_qd = sorted(neo_ids - qd_ids)
    missing_in_neo = sorted(qd_ids - neo_ids)
    rep.check(not missing_in_qd, "Every Neo4j node has a Qdrant vector",
              f"{len(missing_in_qd)} Neo4j nodes have NO vector in Qdrant", missing_in_qd)
    rep.check(not missing_in_neo, "Every Qdrant point has a Neo4j node",
              f"{len(missing_in_neo)} Qdrant points have NO node in Neo4j", missing_in_neo)


def check_point_ids(rep, points):
    rep.section("2. Qdrant point IDs are deterministic (uuid5 of fqn)")
    bad = []
    for p in points:
        fqn = (p.payload or {}).get("fqn")
        if not fqn:
            bad.append(f"point {p.id}: payload has no fqn")
            continue
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, fqn))
        if str(p.id) != expected:
            bad.append(f"{fqn}: id={p.id} expected={expected}")
    rep.check(not bad, "All point IDs match uuid5(fqn)",
              f"{len(bad)} points have an unexpected ID (resume logic would re-embed these)", bad)


def check_fields(rep, neo_nodes, qd_by_fqn):
    rep.section("3. Metadata agreement (Neo4j props vs Qdrant payload)")
    mism = defaultdict(list)
    raw_vs_code = []
    for fqn, n in neo_nodes.items():
        q = qd_by_fqn.get(fqn)
        if q is None:
            continue
        fields = CLASS_FIELDS if n["_kind"] == "Class" else FUNC_FIELDS
        for f in fields:
            a, b = norm(n.get(f)), norm(q.get(f))
            if f == "code_content":
                a, b = digest(a), digest(b)
            if a != b:
                mism[f].append(f"{fqn}  neo4j={short(a)}  qdrant={short(b)}")
        if n["_kind"] == "Class" and q.get("type") != "class":
            mism["type"].append(f"{fqn}  Neo4j label=Class but Qdrant type={q.get('type')!r}")
        if q.get("raw_code") != q.get("code_content"):
            raw_vs_code.append(fqn)

    if not mism:
        rep.ok("All shared fields match for every node present in both stores")
    for field, rows in sorted(mism.items()):
        rep.fail(f"Field '{field}' differs on {len(rows)} nodes", rows)
    if raw_vs_code:
        rep.warn(f"{len(raw_vs_code)} Qdrant payloads have raw_code != code_content", raw_vs_code)


def check_vectors(rep, points, expected_dim):
    rep.section("4. Vector sanity")
    missing, bad_dim, bad_val, zero = [], [], [], []
    for p in points:
        fqn = (p.payload or {}).get("fqn", str(p.id))
        v = p.vector
        if isinstance(v, dict):
            v = next(iter(v.values()), None)
        if not v:
            missing.append(fqn)
            continue
        if len(v) != expected_dim:
            bad_dim.append(f"{fqn}: dim={len(v)}")
        if any(not math.isfinite(x) for x in v):
            bad_val.append(fqn)
        elif not any(v):
            zero.append(fqn)
    rep.check(not missing, "Every point has a vector", f"{len(missing)} points have no vector", missing)
    rep.check(not bad_dim, f"All vectors are {expected_dim}-dimensional",
              f"{len(bad_dim)} vectors have the wrong dimension", bad_dim)
    rep.check(not bad_val, "No NaN/inf values", f"{len(bad_val)} vectors contain NaN/inf", bad_val)
    rep.check(not zero, "No all-zero vectors", f"{len(zero)} vectors are all zeros", zero)


def check_graph_structure(rep, neo, neo_nodes, qd_by_fqn):
    rep.section("5. Neo4j graph structure")
    classes = [n for n in neo_nodes.values() if n["_kind"] == "Class"]
    functions = [n for n in neo_nodes.values() if n["_kind"] == "Function"]

    rep.check(not neo["orphans"], "No Class/Function without a CONTAINS parent",
              f"{len(neo['orphans'])} Class/Function nodes have no parent", sorted(neo["orphans"]))
    rep.check(not neo["no_file"], "Every Class/Function points at an existing File node",
              f"{len(neo['no_file'])} nodes reference a file_path with no File node", sorted(neo["no_file"]))
    if neo["empty_files"]:
        rep.info(f"{len(neo['empty_files'])} File nodes contain no classes/functions "
                 f"(normal for constants-only or script files)")

    # --- CONTAINS: expected vs actual
    file_by_path = {f["file_path"]: f["id"] for f in neo["files"]}
    classes_by_file_name = defaultdict(list)
    for c in classes:
        classes_by_file_name[(c.get("file_path"), c.get("name"))].append(c["id"])

    expected = set()
    for c in classes:
        fid = file_by_path.get(c.get("file_path"))
        if fid:
            expected.add((("File", fid), ("Class", c["id"])))
    for fn in functions:
        parent_cls = fn.get("belongs_to_class")
        if parent_cls is None:
            fid = file_by_path.get(fn.get("file_path"))
            if fid:
                expected.add((("File", fid), ("Function", fn["id"])))
        else:
            for cid in classes_by_file_name.get((fn.get("file_path"), parent_cls), []):
                expected.add((("Class", cid), ("Function", fn["id"])))
    actual = {((r["la"], r["ia"]), (r["lb"], r["ib"])) for r in neo["contains"]}
    diff_edges(rep, "CONTAINS", expected, actual,
               lambda e: f"{e[0][0]}:{e[0][1]} -> {e[1][0]}:{e[1][1]}")

    # --- INHERITS_FROM: expected vs actual
    by_name = defaultdict(list)
    for c in classes:
        by_name[c.get("name")].append(c["id"])
    expected = set()
    for c in classes:
        for base in c.get("inherits_from") or []:
            for pid in by_name.get(base, []):
                expected.add((c["id"], pid))
    actual = {(r["a"], r["b"]) for r in neo["inherits"]}
    diff_edges(rep, "INHERITS_FROM", expected, actual, lambda e: f"{e[0]} -> {e[1]}")

    # --- CALLS: expected (derived from Qdrant payload calls_symbols) vs actual
    symbol_map = defaultdict(set)
    for fqn, payload in qd_by_fqn.items():
        symbol_map[payload.get("name")].add(fqn)
    fn_ids = {fn["id"] for fn in functions}
    expected = set()
    for fqn, payload in qd_by_fqn.items():
        if payload.get("type") not in ("function", "async_function"):
            continue
        for sym in payload.get("calls_symbols") or []:
            for target in symbol_map.get(sym.split(".")[-1], ()):
                if target != fqn and fqn in fn_ids and target in fn_ids:
                    expected.add((fqn, target))
    actual = {(r["a"], r["b"]) for r in neo["calls"]}
    diff_edges(rep, "CALLS (derived from Qdrant calls_symbols)", expected, actual,
               lambda e: f"{e[0]} -> {e[1]}")


def diff_edges(rep, label, expected, actual, fmt):
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    rep.info(f"{label}: {len(actual)} edges in Neo4j, {len(expected)} expected")
    rep.check(not missing, f"{label}: no expected edges missing",
              f"{label}: {len(missing)} expected edges are missing from Neo4j", [fmt(e) for e in missing])
    rep.check(not extra, f"{label}: no unexpected edges",
              f"{label}: {len(extra)} edges in Neo4j that should not exist", [fmt(e) for e in extra])


def check_against_source(rep, src_by_fqn, src_counts, neo_nodes, qd_by_fqn):
    rep.section("6. Both stores vs a fresh parse of the source repo")
    dups = sorted(f for f, c in src_counts.items() if c > 1)
    total_chunks = sum(src_counts.values())
    rep.info(f"Parser produced {total_chunks} chunks / {len(src_by_fqn)} unique fqns")
    if dups:
        rep.warn(f"{len(dups)} fqns are produced by more than one chunk (e.g. @overload, property "
                 f"setters, same-named nested functions). They collapse into ONE node in Neo4j "
                 f"and ONE point in Qdrant, so DB counts are lower than the chunk count.",
                 [f"{f} (x{src_counts[f]})" for f in dups])

    src_ids = set(src_by_fqn)
    for name, ids in (("Neo4j", set(neo_nodes)), ("Qdrant", set(qd_by_fqn))):
        missing = sorted(src_ids - ids)
        extra = sorted(ids - src_ids)
        rep.check(not missing, f"{name}: contains every parsed fqn",
                  f"{name} is missing {len(missing)} fqns the parser found", missing)
        rep.check(not extra, f"{name}: has nothing the parser did not produce (no stale data)",
                  f"{name} has {len(extra)} fqns not in the current source (stale)", extra)

    stale_neo = [f for f, n in neo_nodes.items()
                 if f in src_by_fqn and digest(n.get("code_content")) != digest(src_by_fqn[f]["code_content"])]
    stale_qd = [f for f, p in qd_by_fqn.items()
                if f in src_by_fqn and digest(p.get("code_content")) != digest(src_by_fqn[f]["code_content"])]
    rep.check(not stale_neo, "Neo4j code matches current source",
              f"{len(stale_neo)} Neo4j nodes hold code that differs from the repo", stale_neo)
    rep.check(not stale_qd, "Qdrant code matches current source",
              f"{len(stale_qd)} Qdrant payloads hold code that differs from the repo", stale_qd)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Integrity check: Neo4j vs Qdrant (vs source repo)")
    p.add_argument("--repo-path", default=str(config.REPO_DIR))
    p.add_argument("--skip-source", action="store_true", help="Don't re-parse the repo")
    p.add_argument("--skip-vectors", action="store_true", help="Don't download/check vectors")
    p.add_argument("--max-show", type=int, default=10, help="Max sample rows per issue")
    p.add_argument("--neo4j-uri", default=config.NEO4J_URI)
    p.add_argument("--neo4j-user", default=config.NEO4J_USER)
    p.add_argument("--neo4j-password", default=config.NEO4J_PASSWORD)
    p.add_argument("--neo4j-database", default=config.NEO4J_DATABASE)
    p.add_argument("--qdrant-url", default=config.QDRANT_HOST)
    p.add_argument("--qdrant-api-key", default=config.QDRANT_API_KEY)
    p.add_argument("--qdrant-collection", default=config.QDRANT_COLLECTION_NAME)
    return p.parse_args()


def main():
    args = parse_args()
    rep = Report(args.max_show)

    print("=" * 70)
    print("[*] NEO4J <-> QDRANT INTEGRITY CHECK")
    print("=" * 70)

    print("\nLoading Neo4j...")
    neo = load_neo4j(args)
    print("Loading Qdrant (this scrolls every point)...")
    qd = load_qdrant(args, with_vectors=not args.skip_vectors)

    # Index Neo4j chunk nodes by fqn; detect Class/Function id collisions
    neo_nodes, collisions = {}, []
    for kind, rows in (("Class", neo["classes"]), ("Function", neo["functions"])):
        for r in rows:
            if r["id"] in neo_nodes:
                collisions.append(r["id"])
            neo_nodes[r["id"]] = {**r, "_kind": kind}

    # Index Qdrant payloads by fqn; detect duplicate fqn across points
    qd_by_fqn, dup_qd, seen = {}, [], Counter()
    for p in qd["points"]:
        fqn = (p.payload or {}).get("fqn")
        if fqn is None:
            continue
        seen[fqn] += 1
        qd_by_fqn[fqn] = p.payload
    dup_qd = sorted(f for f, c in seen.items() if c > 1)

    if collisions:
        rep.section("0. Neo4j id collisions")
        rep.fail("Same id used by both a Class and a Function node "
                 "(Qdrant can hold only one point per id)", sorted(collisions))

    check_counts_and_sets(rep, neo_nodes, qd_by_fqn, qd["count"], len(qd["points"]), dup_qd)
    check_point_ids(rep, qd["points"])
    check_fields(rep, neo_nodes, qd_by_fqn)
    if not args.skip_vectors:
        check_vectors(rep, qd["points"], qd["dim"])
    check_graph_structure(rep, neo, neo_nodes, qd_by_fqn)

    if not args.skip_source:
        repo = Path(args.repo_path)
        if repo.exists():
            _, src_by_fqn, src_counts = load_source(repo)
            check_against_source(rep, src_by_fqn, src_counts, neo_nodes, qd_by_fqn)
        else:
            rep.section("6. Source comparison")
            rep.warn(f"Repo path not found, skipped: {repo}")

    print("\n" + "=" * 70)
    verdict = "PASS" if rep.failures == 0 else "FAIL"
    print(f"[*] RESULT: {verdict}   failures={rep.failures}   warnings={rep.warnings}")
    print("=" * 70)
    sys.exit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()