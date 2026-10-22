"""
graph_traversal.py – Graph Expansion for Codebase RAG

Strategy:
  1. Fetch strict Top-k results from Qdrant (no similarity cutoff – Neo4j handles structural rescue).
  2. Pass ALL seed node IDs into Neo4j in a single batched query (WHERE seed.id IN $seed_ids).
  3. Traverse 1–2 hops over explicit uppercase relationship types: CALLS, INHERITS_FROM, CONTAINS.
  4. Return adjacent nodes with name, file_path, type, and hop distance.
"""
import os
import sys
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, DatabaseUnavailable, SessionExpired

try:
    from config import (
        NEO4J_URI as CONFIG_NEO4J_URI,
        NEO4J_USER as CONFIG_NEO4J_USER,
        NEO4J_PASSWORD as CONFIG_NEO4J_PASSWORD,
        NEO4J_DATABASE as CONFIG_NEO4J_DATABASE,
    )
except ImportError:
    CONFIG_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    CONFIG_NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    CONFIG_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    CONFIG_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

from retrieval.retrieve import Retrieval

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  GraphTraversal Class
# --------------------------------------------------------------------------- #
class GraphTraversal:
    """
    Connects to Neo4j to expand Qdrant seed nodes by traversing 1 to 2 hops
    across graph relationships (:CALLS, :CONTAINS, :INHERITS_FROM).
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
        retriever: Optional[Any] = None,
    ):
        self.neo4j_uri = neo4j_uri or CONFIG_NEO4J_URI
        self.neo4j_user = neo4j_user or CONFIG_NEO4J_USER
        self.neo4j_password = neo4j_password or CONFIG_NEO4J_PASSWORD
        self.neo4j_database = neo4j_database or CONFIG_NEO4J_DATABASE
        self.retriever = retriever
        self.driver: Optional[AsyncDriver] = None

    async def connect(self):
        """Initialize connection to Neo4j if not already connected."""
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            logger.info(f"[GraphTraversal] Connected to Neo4j at {self.neo4j_uri}")

    async def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None

    async def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Fetch node details from Neo4j for a given node_id."""
        if not self.driver:
            await self.connect()

        query = """
        MATCH (n {id: $node_id})
        RETURN
            n.id AS id,
            labels(n)[0] AS type,
            n.name AS name,
            n.file_path AS file_path,
            n.belongs_to_class AS belongs_to_class
        LIMIT 1
        """
        async with self.driver.session(database=self.neo4j_database) as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            if record:
                return dict(record)
        return None

    async def get_adjacent_nodes_batch(
        self,
        seed_ids: List[str],
        max_hops: int = 2,
        per_seed_limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Production-grade batched Neo4j traversal — single round-trip for all seeds.

        Design decisions:
          1. UNWIND $seed_ids vectorises the lookup — no per-node round-trips.
          2. Outer CALL { WITH sid ... LIMIT $per_seed_limit } is a correlated
             subquery that enforces a hard cardinality cap *per seed* before
             results are merged (prevents hub-node flooding).
          3. Inner CALL { ... UNION ALL ... } handles each relationship type with
             its explicit directionality policy:
               - CONTAINS    : undirected — walk UP to parent File/Class OR
                               DOWN to child Class/Function.
               - CALLS (out) : (seed)-[:CALLS*1..2]->(n) — find what this node
                               calls (callees / dependencies).
               - CALLS (in)  : (seed)<-[:CALLS*1..2]-(n) — find who calls this
                               node (callers / upstream consumers). Critical when
                               the seed is a utility/helper that has no outgoing
                               calls of its own.
               - INHERITS_FROM: undirected — find base class OR subclasses.

        Returns list of dicts with:
          seed_id, id, type, name, file_path, belongs_to_class, hops,
          relationship_types
        """
        if not self.driver:
            await self.connect()

        # Single undirected MATCH handles all directions in one pass:
        #   (seed)-[:CALLS]-(adj)          matches callers AND callees
        #   (seed)-[:CONTAINS]-(adj)       matches parent File/Class AND child nodes
        #   (seed)-[:INHERITS_FROM]-(adj)  matches base class AND subclasses
        # Outer CALL subquery enforces the per-seed cardinality cap.
        safe_max_hops = max(1, min(max_hops, 3))
        query = f"""
        UNWIND $seed_ids AS sid
        CALL (sid) {{
            MATCH (seed {{id: sid}})
            MATCH path = (seed)-[:CALLS|INHERITS_FROM|CONTAINS*1..{safe_max_hops}]-(adj)
            WHERE adj.id <> seed.id
            WITH adj,
                 min(length(path))  AS min_hops,
                 head(collect(path)) AS sample_path
            RETURN adj, min_hops, sample_path
            ORDER BY min_hops ASC, adj.name ASC
            LIMIT $per_seed_limit
        }}
        RETURN
            sid                                                        AS seed_id,
            adj.id                                                     AS id,
            labels(adj)[0]                                             AS type,
            adj.name                                                   AS name,
            adj.file_path                                              AS file_path,
            adj.belongs_to_class                                       AS belongs_to_class,
            min_hops                                                   AS hops,
            [rel IN relationships(sample_path) | type(rel)]           AS relationship_types
        ORDER BY seed_id ASC, hops ASC, type ASC, name ASC
        """
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with self.driver.session(database=self.neo4j_database) as session:
                    result = await session.run(
                        query,
                        seed_ids=seed_ids,
                        per_seed_limit=per_seed_limit,
                    )
                    records = await result.data()
                    return records
            except (ServiceUnavailable, DatabaseUnavailable, SessionExpired) as exc:
                if attempt == max_retries:
                    logger.error(f"[GraphTraversal] Neo4j connection failed after {max_retries} attempts: {exc}")
                    raise
                wait_time = attempt * 2
                logger.warning(
                    f"[GraphTraversal] Transient Neo4j error ({exc}). Reconnecting in {wait_time}s (attempt {attempt}/{max_retries})..."
                )
                await asyncio.sleep(wait_time)
                await self.close()
                await self.connect()
        return []

    async def traverse_top_seeds(
        self,
        qdrant_results: List[Dict[str, Any]],
        top_seeds: int = 2,
        max_hops: int = 2,
        per_seed_limit: int = 20,
        retriever: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Passes ALL top `top_seeds` Qdrant results into Neo4j as a single batched
        traversal query (UNWIND $seed_ids), 1–2 hops over
        CALLS / INHERITS_FROM / CONTAINS (undirected).

        Collects unique IDs from Neo4j graph expansion that were not in Qdrant's
        initial seeds, batch-retrieves their payloads from Qdrant, and merges
        the structural expansion results with their code bodies.

        Returns list of seed blocks containing structural graph info and complete code snippets.
        """
        seeds_to_process = qdrant_results[:top_seeds]
        if not seeds_to_process:
            return []

        # 1. Build seed metadata map keyed by node id
        seed_meta: Dict[str, Dict[str, Any]] = {}
        for seed in seeds_to_process:
            sid = str(seed.get("id") or seed.get("payload", {}).get("id", ""))
            sp = seed.get("payload", {})
            seed_meta[sid] = {
                "seed_id": sid,
                "seed_name": seed.get("name") or sp.get("name") or "Unknown",
                "seed_file_path": sp.get("file_path") or "Unknown",
                "similarity_score": seed.get("score", 0.0),
                "payload": sp,
                "raw_code": sp.get("raw_code") or sp.get("code_content", ""),
                "start_line": sp.get("start_line"),
                "end_line": sp.get("end_line"),
                "decorators": sp.get("decorators", []),
                "is_api_endpoint": sp.get("is_api_endpoint", False),
                "adjacent_nodes": [],
            }

        # 2. Single batched Neo4j round-trip — per-seed cardinality enforced inside
        all_adjacent = []
        try:
            all_adjacent = await self.get_adjacent_nodes_batch(
                seed_ids=list(seed_meta.keys()),
                max_hops=max_hops,
                per_seed_limit=per_seed_limit,
            )
        except Exception as exc:
            logger.warning(
                f"[GraphTraversal] Neo4j graph expansion unavailable ({exc}). Proceeding with primary vector seeds only."
            )
            all_adjacent = []

        # 3. Collect unique IDs returned from Neo4j that weren't already fetched from Qdrant
        seed_ids = set(seed_meta.keys())
        adjacent_ids_to_fetch = list(dict.fromkeys(
            str(row["id"]) for row in all_adjacent
            if row.get("id") and str(row["id"]) not in seed_ids
        ))

        # 4. Batch-retrieve payloads from Qdrant using client.retrieve(ids=[...])
        active_retriever = retriever or self.retriever
        adj_payloads: Dict[str, Dict[str, Any]] = {}
        if active_retriever and adjacent_ids_to_fetch:
            adj_payloads = await active_retriever.get_payloads_by_ids(adjacent_ids_to_fetch)

        # 5. Merge the structural expansion results with their code bodies
        for row in all_adjacent:
            nid = str(row.get("id"))
            p = adj_payloads.get(nid) or (seed_meta[nid].get("payload", {}) if nid in seed_meta else {})
            row["payload"] = p
            row["raw_code"] = p.get("raw_code") or p.get("code_content", "")
            row["start_line"] = p.get("start_line")
            row["end_line"] = p.get("end_line")
            row["decorators"] = p.get("decorators", [])
            row["is_api_endpoint"] = p.get("is_api_endpoint", False)

        # 6. Group adjacent rows back to their originating seed
        for row in all_adjacent:
            sid = row.get("seed_id")
            if sid in seed_meta:
                seed_meta[sid]["adjacent_nodes"].append(row)

        # 7. Preserve Qdrant ranking order
        expanded = []
        for seed in seeds_to_process:
            sid = str(seed.get("id") or seed.get("payload", {}).get("id", ""))
            block = seed_meta[sid]
            block["adjacent_count"] = len(block["adjacent_nodes"])
            expanded.append(block)

        return expanded


# --------------------------------------------------------------------------- #
#  Context Window Builder
# --------------------------------------------------------------------------- #
def build_context_window(
    expanded_results: List[Dict[str, Any]],
    max_chars: int = 10000,
    max_seed_snippet_chars: int = 1500,
    max_adj_snippet_chars: int = 1000,
    max_adj_code_count: int = 2,
) -> str:
    """
    Builds a strictly budgeted prompt context window combining:
      1. Primary vector seed chunks with similarity scores and capped code snippets.
      2. Top 1-hop direct dependencies with code snippets (capped count & size).
      3. 2-hop connected nodes as compact structural references (no code bodies).
    Enforces a strict ~10,000 character limit (~2,500 tokens) to guarantee
    compliance with strict 8,000 TPM limits on large models like openai/gpt-oss-120b.
    """
    if not expanded_results:
        return "No relevant codebase context found."

    seen_code_ids = set()
    current_length = 0
    adj_code_snippets_added = 0

    sections: List[str] = []
    sections.append("# ====================================================================")
    sections.append("# CODEBASE RETRIEVAL CONTEXT (Vector Seeds + Graph Traversal Expansion)")
    sections.append("# ====================================================================\n")

    # Section 1: Top Vector Seeds (Primary Entry Points)
    sections.append("## PRIMARY RELEVANT SEEDS (Vector Search Matches)")
    for idx, seed in enumerate(expanded_results, 1):
        sid = seed.get("seed_id", "")
        seen_code_ids.add(sid)
        name = seed.get("seed_name", "Unknown")
        file_path = seed.get("seed_file_path", "Unknown")
        score = seed.get("similarity_score", 0.0)
        s_line = seed.get("start_line")
        e_line = seed.get("end_line")
        line_info = f" (Lines {s_line}-{e_line})" if s_line and e_line else ""
        raw_code = seed.get("raw_code", "").strip()
        if len(raw_code) > max_seed_snippet_chars:
            raw_code = raw_code[:max_seed_snippet_chars].rstrip() + "\n# ... [Remaining code omitted to stay within token budget] ..."

        sections.append(f"### [Seed #{idx}] {name}")
        sections.append(f"- File: {file_path}{line_info}")
        sections.append(f"- Similarity Score: {score:.4f}")
        sections.append(f"- Node ID: {sid}")
        if raw_code:
            sections.append("```python")
            sections.append(raw_code)
            sections.append("```\n")
        else:
            sections.append("*(No code body attached)*\n")

    current_length = sum(len(s) + 1 for s in sections)

    # Section 2: Structural Graph Expansion (1-2 Hops)
    sections.append("## STRUCTURAL GRAPH EXPANSION (1-2 Hops via Neo4j)")
    for idx, seed in enumerate(expanded_results, 1):
        s_name = seed.get("seed_name", "Unknown")
        adj_nodes = seed.get("adjacent_nodes", [])
        if not adj_nodes:
            continue

        # Sort so 1-hop comes before 2-hop
        sorted_adj = sorted(adj_nodes, key=lambda n: n.get("hops", 1))

        sections.append(f"### Graph Connections for Seed #{idx} ({s_name}):")
        for node in sorted_adj:
            nid = node.get("id", "")
            ntype = node.get("type", "Node")
            name = node.get("name", "Unknown")
            file_path = node.get("file_path", "Unknown")
            hops = node.get("hops", 1)
            rels = " -> ".join(node.get("relationship_types", []))
            s_line = node.get("start_line")
            e_line = node.get("end_line")
            line_info = f" (Lines {s_line}-{e_line})" if s_line and e_line else ""
            raw_code = node.get("raw_code", "").strip()

            line_header = f"- [{hops}-Hop via {rels}] {ntype} `{name}` ({file_path}{line_info})"
            sections.append(line_header)
            current_length += len(line_header) + 1

            if nid in seen_code_ids:
                note = f"  *(Code body for `{name}` already included above)*"
                sections.append(note)
                current_length += len(note) + 1
            elif raw_code and hops == 1 and adj_code_snippets_added < max_adj_code_count:
                # Add 1-hop code snippet only if within strict budget
                if len(raw_code) > max_adj_snippet_chars:
                    raw_code = raw_code[:max_adj_snippet_chars].rstrip() + "\n# ... [Snippet truncated] ..."
                code_block_len = len(raw_code) + 25
                if (current_length + code_block_len) < max_chars:
                    seen_code_ids.add(nid)
                    adj_code_snippets_added += 1
                    sections.append("  ```python")
                    sections.append("  " + raw_code.replace("\n", "\n  "))
                    sections.append("  ```")
                    current_length += code_block_len
            sections.append("")

    return "\n".join(sections)


# --------------------------------------------------------------------------- #
#  Pretty Print Helper
# --------------------------------------------------------------------------- #
def print_traversal_results(traversal_results: List[Dict[str, Any]], show_context: bool = False):
    """Prints formatted output of seed nodes and their 1..2 hop adjacent nodes with code snippet stats."""
    print("\n" + "=" * 70)
    print("  GRAPH EXPANSION RESULTS (1 to 2 Hops from Top-2 Qdrant Seeds)")
    print("=" * 70)

    if not traversal_results:
        print("[INFO] No seed nodes provided for graph expansion.")
        return

    for idx, item in enumerate(traversal_results, 1):
        s_code = item.get("raw_code", "")
        code_tag = f" | [code: {len(s_code)} chars]" if s_code else ""
        s_line = item.get("start_line")
        e_line = item.get("end_line")
        lines_tag = f" (Lines {s_line}-{e_line})" if s_line and e_line else ""

        print(f"\n[Seed #{idx}] {item['seed_name']}{lines_tag}{code_tag}")
        print(f"  Similarity Score : {item['similarity_score']:.4f}")
        print(f"  File Path        : {item['seed_file_path']}")
        print(f"  Node ID (UUID5)  : {item['seed_id']}")
        print(f"  Adjacent Nodes   : {item['adjacent_count']} found (1-2 hops)")
        print("  " + "-" * 66)

        adj_nodes = item["adjacent_nodes"]
        if not adj_nodes:
            print("    (No adjacent nodes found in Neo4j)")
            continue

        # Group by 1-hop vs 2-hop
        hop1 = [n for n in adj_nodes if n["hops"] == 1]
        hop2 = [n for n in adj_nodes if n["hops"] == 2]

        if hop1:
            print(f"  [*] 1-Hop Directly Connected ({len(hop1)} nodes):")
            for node in hop1:
                rel_str = " -> ".join(node.get("relationship_types", []))
                n_code = node.get("raw_code", "")
                c_tag = f" [code: {len(n_code)} chars]" if n_code else " (structural)"
                n_s = node.get("start_line")
                n_e = node.get("end_line")
                l_tag = f":L{n_s}-L{n_e}" if n_s and n_e else ""
                print(f"      -> [{node['type']}] {node['name']} ({node['file_path']}{l_tag}) via [{rel_str}]{c_tag}")

        if hop2:
            print(f"  [*] 2-Hop Connected ({len(hop2)} nodes):")
            for node in hop2:
                rel_str = " -> ".join(node.get("relationship_types", []))
                n_code = node.get("raw_code", "")
                c_tag = f" [code: {len(n_code)} chars]" if n_code else " (structural)"
                n_s = node.get("start_line")
                n_e = node.get("end_line")
                l_tag = f":L{n_s}-L{n_e}" if n_s and n_e else ""
                print(f"      -> [{node['type']}] {node['name']} ({node['file_path']}{l_tag}) via [{rel_str}]{c_tag}")

    if show_context:
        print("\n" + "=" * 70)
        print("  COMPILED LLM CONTEXT WINDOW PREVIEW")
        print("=" * 70)
        context_str = build_context_window(traversal_results)
        print(context_str)


# --------------------------------------------------------------------------- #
#  CLI / Interactive Main
# --------------------------------------------------------------------------- #
async def main():
    parser = argparse.ArgumentParser(description="Graph Traversal (1-2 Hops) for Top-2 Qdrant Seeds")
    parser.add_argument("--query", type=str, default=None, help="Search query to run against Qdrant")
    parser.add_argument("--node-id", type=str, default=None, help="Explicit node ID to traverse in Neo4j")
    parser.add_argument("--top-seeds", type=int, default=2, help="Number of top Qdrant seeds to traverse (default: 2)")
    parser.add_argument("--max-hops", type=int, default=2, help="Max hops in Neo4j (default: 2)")
    parser.add_argument("--show-context", action="store_true", help="Print the full compiled LLM prompt context window")
    args = parser.parse_args()

    graph = GraphTraversal()
    retriever = Retrieval()

    try:
        await graph.connect()

        # If a single explicit node-id is provided via CLI
        if args.node_id:
            node_info = await graph.get_node_by_id(args.node_id)
            seed_name = node_info.get("name", "Unknown") if node_info else "Unknown"
            seed_file = node_info.get("file_path", "Unknown") if node_info else "Unknown"
            adj = await graph.get_adjacent_nodes(args.node_id, max_hops=args.max_hops)
            print_traversal_results([{
                "seed_id": args.node_id,
                "seed_name": seed_name,
                "seed_file_path": seed_file,
                "similarity_score": 1.0,
                "adjacent_count": len(adj),
                "adjacent_nodes": adj,
            }], show_context=args.show_context)
            return

        await retriever.init_clients()

        # If query is passed via CLI flag
        if args.query:
            # Strict Top-k – no score cutoff, Neo4j handles structural rescue
            q_results = await retriever.retrieve(args.query, limit=args.top_seeds, score_threshold=None)
            expanded = await graph.traverse_top_seeds(
                qdrant_results=q_results,
                top_seeds=args.top_seeds,
                max_hops=args.max_hops,
                retriever=retriever,
            )
            print_traversal_results(expanded, show_context=args.show_context)
            return

        # Interactive loop
        print("\n" + "=" * 70)
        print("  GRAPH TRAVERSAL (1-2 Hops | Batch Seed Expansion | CALLS/INHERITS_FROM/CONTAINS)")
        print("  Type 'exit' to quit")
        print("=" * 70)

        query_history: List[str] = []
        while True:
            prompt = "\nEnter your search query (or 'exit' to quit): " if not query_history else f"\nEnter follow-up query [turn {len(query_history)+1}] (or 'exit'): "
            query = input(prompt).strip()

            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting graph traversal session.")
                break

            effective_query = query
            if query_history:
                effective_query = await retriever.contextualize_query(query, query_history)
                if effective_query != query:
                    print(f"[Retrieval] Contextualized query: '{effective_query}'")

            print(f"\n[1/2] Retrieving strict Top-{args.top_seeds} vector matches from Qdrant (no cutoff)...")
            # No score_threshold – fetch strict Top-k unconditionally
            q_results = await retriever.retrieve(effective_query, limit=args.top_seeds, score_threshold=None)

            if not q_results:
                print("[INFO] No chunks returned from Qdrant.")
                continue

            print(f"[2/2] Batch-expanding {len(q_results)} seed(s) in Neo4j & merging code payloads from Qdrant...")
            expanded = await graph.traverse_top_seeds(
                qdrant_results=q_results,
                top_seeds=args.top_seeds,
                max_hops=args.max_hops,
                retriever=retriever,
            )
            print_traversal_results(expanded, show_context=args.show_context)
            query_history.append(effective_query)

    finally:
        await graph.close()
        await retriever.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
