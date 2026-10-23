"""Durable library catalog and scoped dense/lexical retrieval.

SQLite is authoritative: vectors are published before a catalog transaction
marks a document ready. Search filters by the published generation, so failed
uploads, interrupted retries and failed vector cleanup cannot expose stale data.
"""

import hashlib
import math
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import unicodedata
import uuid
from pathlib import Path

from chatbot.log import logger
from rag_chatbot.schemas import Chunk, Evidence


_UNSAFE_SOURCE_NAME = re.compile(r'[<>:"/\\|?*]')
_RESERVED_SOURCE_NAME = re.compile(
    r"^(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9\u00b9\u00b2\u00b3]|LPT[1-9\u00b9\u00b2\u00b3])(?:\.|$)",
    re.IGNORECASE,
)


def validate_display_name(path, display_name):
    """Return a safe explicit basename, or None when no override was requested.

    Display names are catalog metadata. The uploaded original and UUID-managed
    copy keep their paths and extensions regardless of the chosen label.
    """
    if display_name is None:
        return None
    if not isinstance(display_name, str):
        raise ValueError("Source names must be text.")
    if any(unicodedata.category(character) == "Cc" for character in display_name):
        raise ValueError("Source names cannot contain control characters.")
    if not display_name.strip():
        return None
    name = display_name
    if (name != name.strip() or name.endswith(".") or _UNSAFE_SOURCE_NAME.search(name)
            or _RESERVED_SOURCE_NAME.match(name.split(".", 1)[0].rstrip(" "))):
        raise ValueError("Source names must be safe filenames without path separators, control characters, "
                         "reserved Windows names, or leading/trailing spaces or trailing dots.")
    suffix = Path(path).suffix
    requested_suffix = Path(name).suffix
    if requested_suffix and requested_suffix.casefold() != suffix.casefold():
        raise ValueError(f"Keep the original file extension {suffix or '(no extension)'!r} in the source name.")
    if not requested_suffix:
        name += suffix
    if len(name.encode("utf-16-le")) // 2 > 255:
        raise ValueError("Source names are limited to 255 characters, including the file extension.")
    return name


class LibraryStore:
    def __init__(self, settings, *, vector_client=None):
        self.settings = settings
        self.data_dir = Path(settings.data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._managed_path("sources").mkdir(exist_ok=True)
        self._vectors = vector_client
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self._managed_path("library.sqlite3"), timeout=30,
                                   check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
                collection TEXT NOT NULL, content_hash TEXT NOT NULL,
                fingerprint TEXT NOT NULL, dimensions INTEGER NOT NULL,
                status TEXT NOT NULL, error TEXT NOT NULL DEFAULT '',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                generation TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS document_identity
                ON documents(content_hash, collection, fingerprint)
                WHERE status != 'deleted';
            CREATE INDEX IF NOT EXISTS document_scope
                ON documents(collection, fingerprint, status);
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
                text TEXT NOT NULL, locator TEXT NOT NULL, kind TEXT NOT NULL,
                image_path TEXT, ordinal INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, content='chunks', content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 0'
            );
            CREATE TRIGGER IF NOT EXISTS chunks_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid,text) VALUES (new.rowid,new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_delete AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts,rowid,text)
                    VALUES ('delete',old.rowid,old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_update AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts,rowid,text)
                    VALUES ('delete',old.rowid,old.text);
                INSERT INTO chunks_fts(rowid,text) VALUES (new.rowid,new.text);
            END;
            CREATE TEMP TABLE scoped_documents (id TEXT PRIMARY KEY);
        """)
        # Preserve extraction order for whole-document tools. Older libraries
        # inserted chunks in order, so rowid provides the migration ordering.
        with self._db:
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(chunks)")}
            if "ordinal" not in columns:
                self._db.execute("ALTER TABLE chunks ADD COLUMN ordinal INTEGER NOT NULL DEFAULT 0")
                self._db.execute("""
                    WITH ordered AS (
                        SELECT id, ROW_NUMBER() OVER (
                            PARTITION BY document_id ORDER BY rowid) - 1 AS position FROM chunks
                    )
                    UPDATE chunks SET ordinal=(SELECT position FROM ordered WHERE ordered.id=chunks.id)
                """)
            self._db.execute("CREATE INDEX IF NOT EXISTS chunks_order ON chunks(document_id,ordinal)")

    def _managed_path(self, value):
        path = Path(value)
        path = (self.data_dir / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.data_dir):
            raise ValueError("Library paths must stay inside the configured data directory.")
        return path

    @staticmethod
    def _label(value, name):
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise ValueError(f"{name} must be a non-empty string of at most 512 characters.")
        if any(ord(char) < 32 for char in value):
            raise ValueError(f"{name} cannot contain control characters.")
        return value.strip()

    @staticmethod
    def _hash_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _collection_name(self, fingerprint, dimensions):
        identity = f"{fingerprint}:{dimensions}"
        if self.settings.namespace:
            identity = f"{self.settings.namespace}:{identity}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:32]
        return f"rag_{digest}"

    @property
    def vector_client(self):
        if self._vectors is None:
            from qdrant_client import QdrantClient
            local_path = getattr(self.settings, "qdrant_local_path", None)
            if local_path:
                self._vectors = QdrantClient(path=str(Path(local_path).resolve()))
            else:
                self._vectors = QdrantClient(url=self.settings.qdrant_url,
                                             api_key=self.settings.qdrant_api_key, timeout=15)
        return self._vectors

    def ensure_index(self, fingerprint=None, dimensions=None):
        """Check Qdrant/create its collection before spending tokens on embeddings."""
        from qdrant_client import models
        fingerprint = fingerprint or self.settings.fingerprint()
        dimensions = dimensions or self.settings.embedding_dimensions
        name = self._collection_name(fingerprint, dimensions)
        with self._lock:
            client = self.vector_client
            if not client.collection_exists(name):
                client.create_collection(name, vectors_config=models.VectorParams(
                    size=dimensions, distance=models.Distance.COSINE))
                # Local Qdrant used in tests ignores payload indexes. The server
                # needs them for large libraries and scoped vector search.
                from qdrant_client.local.qdrant_local import QdrantLocal
                if not isinstance(client._client, QdrantLocal):
                    for field in ("document_id", "collection", "generation"):
                        client.create_payload_index(name, field,
                                                    field_schema=models.PayloadSchemaType.KEYWORD,
                                                    wait=True)
            info = client.get_collection(name)
            if info.config.params.vectors.size != dimensions:
                raise ValueError("Qdrant embedding dimensions do not match the library settings.")
        return name

    def check_connection(self):
        return self.ensure_index()

    def _document(self, row):
        if row is None:
            return None
        result = dict(row)
        result["path"] = str(self._managed_path(result["path"]))
        return result

    def get_document(self, document_id):
        with self._lock:
            row = self._db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
            return self._document(row)

    def list_documents(self, collection=None):
        with self._lock:
            sql = "SELECT * FROM documents WHERE status != 'deleted'"
            values = []
            if collection is not None:
                sql += " AND collection=?"
                values.append(collection)
            rows = self._db.execute(sql + " ORDER BY created_at, name, id", values).fetchall()
            return [self._document(row) for row in rows]

    def collections(self):
        with self._lock:
            return [row[0] for row in self._db.execute(
                "SELECT DISTINCT collection FROM documents WHERE status != 'deleted' ORDER BY collection"
            )]

    def stage_document(self, path, collection, fingerprint, *, force=False, display_name=None):
        display_name = validate_display_name(path, display_name)
        source = Path(path).resolve()
        if not source.is_file():
            raise ValueError("Choose an existing document file to import.")
        collection = self._label(collection, "Collection")
        fingerprint = self._label(fingerprint, "Index fingerprint")
        digest = self._hash_file(source)
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            existing = self._db.execute(
                "SELECT * FROM documents WHERE content_hash=? AND collection=? "
                "AND fingerprint=? AND status != 'deleted'", (digest, collection, fingerprint)
            ).fetchone()
            if existing is not None:
                destination = self._managed_path(existing["path"])
                # A catalog match does not guarantee that the managed original
                # survived a restore, interrupted copy, or manual file change.
                # Repair it from this verified upload without redoing paid work.
                if (not destination.is_file() or self._hash_file(destination) != digest):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    descriptor, temporary_name = tempfile.mkstemp(prefix=".restore-", dir=destination.parent)
                    os.close(descriptor)
                    temporary = Path(temporary_name)
                    try:
                        shutil.copyfile(source, temporary)
                        if self._hash_file(temporary) != digest:
                            raise ValueError("The document changed during import. Please try again.")
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                if display_name is not None and display_name != existing["name"]:
                    self._db.execute("UPDATE documents SET name=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                     (display_name, existing["id"]))
                if existing["status"] == "ready" and existing["generation"] and not force:
                    return dict(self.get_document(existing["id"]), unchanged=True)
                # A staging row can also be retried after a process interruption.
                # Publication later checks its status again before making it ready.
                self._db.execute("UPDATE documents SET status='staging',error='',"
                                 "updated_at=CURRENT_TIMESTAMP WHERE id=?", (existing["id"],))
                return dict(self.get_document(existing["id"]), unchanged=False)
            document_id = str(uuid.uuid4())
            suffix = source.suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,16}", source.suffix) else ".bin"
            destination = self._managed_path(Path("sources") / document_id / f"original{suffix}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(source, destination)
                if self._hash_file(destination) != digest:
                    raise ValueError("The document changed during import. Please try again.")
                self._db.execute(
                    "INSERT INTO documents(id,name,path,collection,content_hash,fingerprint,dimensions,status) "
                    "VALUES (?,?,?,?,?,?,?,'staging')",
                    (document_id, display_name or source.name, str(destination.relative_to(self.data_dir)),
                     collection, digest, fingerprint, self.settings.embedding_dimensions),
                )
            except Exception:
                # Cleanup only our newly created file; preserve the original error.
                destination.unlink(missing_ok=True)
                raise
            return dict(self.get_document(document_id), unchanged=False)

    def _vector(self, vector, dimensions):
        values = [float(value) for value in vector]
        if len(values) != dimensions or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Each embedding must have {dimensions} finite values.")
        if not any(values):
            raise ValueError("An embedding cannot be an all-zero vector.")
        return values

    def commit_document(self, document_id, chunks, vectors):
        from qdrant_client import models
        chunks, vectors = list(chunks), list(vectors)
        with self._lock:
            document = self.get_document(document_id)
            if document is None or document["status"] != "staging":
                raise ValueError("Only a staged document can be published.")
            try:
                if not chunks or len(chunks) != len(vectors):
                    raise ValueError("Document chunks and embeddings must be non-empty and have matching lengths.")
                if len({chunk.id for chunk in chunks}) != len(chunks):
                    raise ValueError("Chunk identifiers must be unique.")
                for chunk in chunks:
                    if chunk.document_id != document_id or not chunk.id or not chunk.text.strip():
                        raise ValueError("Each non-empty chunk must belong to the staged document.")
                    if chunk.image_path:
                        self._managed_path(chunk.image_path)
                    owner = self._db.execute("SELECT document_id FROM chunks WHERE id=?", (chunk.id,)).fetchone()
                    if owner and owner[0] != document_id:
                        raise ValueError("A chunk identifier already belongs to another document.")
                vectors = [self._vector(vector, document["dimensions"]) for vector in vectors]
                name = self.ensure_index(document["fingerprint"], document["dimensions"])
                generation = str(uuid.uuid4())
                for offset in range(0, len(chunks), 128):
                    points = [models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag:{document_id}:{chunk.id}:{generation}")),
                        vector=vector, payload={"chunk_id": chunk.id, "document_id": document_id,
                                                "collection": document["collection"],
                                                "generation": generation},
                    ) for chunk, vector in zip(chunks[offset:offset + 128], vectors[offset:offset + 128])]
                    self.vector_client.upsert(name, points=points, wait=True)
                with self._db:
                    self._db.execute("BEGIN IMMEDIATE")
                    status = self._db.execute("SELECT status FROM documents WHERE id=?", (document_id,)).fetchone()
                    if status is None or status[0] != "staging":
                        raise ValueError("The document was deleted or published during indexing.")
                    self._db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
                    self._db.executemany(
                        "INSERT INTO chunks(id,document_id,text,locator,kind,image_path,ordinal) VALUES (?,?,?,?,?,?,?)",
                        [(chunk.id, document_id, chunk.text, chunk.locator, chunk.kind,
                          str(self._managed_path(chunk.image_path)) if chunk.image_path else None, ordinal)
                         for ordinal, chunk in enumerate(chunks)],
                    )
                    self._db.execute(
                        "UPDATE documents SET status='ready',error='',chunk_count=?,generation=?,"
                        "updated_at=CURRENT_TIMESTAMP WHERE id=?", (len(chunks), generation, document_id))
            except Exception as exc:
                # Publishing spans two stores. Any error leaves catalog visibility
                # off, including errors after Qdrant accepted some or all vectors.
                self.fail_document(document_id, str(exc))
                raise
            return self.get_document(document_id)

    def fail_document(self, document_id, error):
        with self._lock, self._db:
            self._db.execute("UPDATE documents SET status='failed',error=?,updated_at=CURRENT_TIMESTAMP "
                             "WHERE id=? AND status='staging'", (str(error)[:4000], document_id))

    def delete_document(self, document_id):
        from qdrant_client import models
        with self._lock:
            document = self.get_document(document_id)
            if document is None:
                raise ValueError("The document does not exist.")
            with self._db:
                self._db.execute("UPDATE documents SET status='deleted',chunk_count=0,"
                                 "updated_at=CURRENT_TIMESTAMP WHERE id=?", (document_id,))
                self._db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            try:
                name = self._collection_name(document["fingerprint"], document["dimensions"])
                if self.vector_client.collection_exists(name):
                    self.vector_client.delete(name, points_selector=models.FilterSelector(
                        filter=models.Filter(must=[models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id))])), wait=True)
            except Exception as exc:
                # Tombstones, not successful vector cleanup, control visibility.
                logger.warning("Document %s was removed; vector cleanup can be retried: %s", document_id, exc)
            # Originals are retained so previously saved citations remain usable.
            return self.get_document(document_id)

    def get_chunk(self, chunk_id):
        with self._lock:
            row = self._db.execute(
                "SELECT c.* FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE c.id=? AND d.status='ready' AND d.fingerprint=? AND d.generation!=''",
                (chunk_id, self.settings.fingerprint()),
            ).fetchone()
            return Chunk(**{key: row[key] for key in Chunk.__dataclass_fields__}) if row else None

    def snapshot_sources(self, document_ids, collection=None, *, max_documents=40,
                         max_chunks=512, max_characters=500_000):
        """Snapshot a complete, bounded selection without querying Qdrant.

        A single SQLite statement sees one published catalog generation. Reject
        stale selections rather than quietly generating an incomplete artifact.
        The caller owns the returned copies and must validate them before use.
        """
        if (not isinstance(document_ids, (list, tuple))
                or any(not isinstance(value, str) or not value for value in document_ids)):
            raise ValueError("Select source document identifiers as a list.")
        selected = list(dict.fromkeys(document_ids))
        if not selected:
            raise ValueError("Select at least one indexed source for this tool.")
        if len(selected) > max_documents:
            raise ValueError(f"Select at most {max_documents} documents per tool run.")
        if collection is not None:
            collection = self._label(collection, "Collection")
        placeholders = ",".join("?" for _ in selected)
        query = (
            "SELECT d.*,c.id AS chunk_id,c.text AS chunk_text,c.locator AS chunk_locator,"
            "c.image_path AS chunk_image_path,c.ordinal AS chunk_ordinal "
            "FROM documents d JOIN chunks c ON c.document_id=d.id "
            f"WHERE d.id IN ({placeholders}) AND d.status='ready' AND d.fingerprint=? "
            "AND d.generation!=''"
        )
        values = [*selected, self.settings.fingerprint()]
        if collection is not None:
            query += " AND d.collection=?"
            values.append(collection)
        query += " ORDER BY d.id,c.ordinal,c.rowid LIMIT ?"
        values.append(max_chunks + 1)
        with self._lock:
            rows = self._db.execute(query, values).fetchall()
            if len(rows) > max_chunks:
                raise ValueError(f"This selection exceeds {max_chunks} chunks. Select fewer sources.")
            if sum(len(row["chunk_text"]) for row in rows) > max_characters:
                raise ValueError("This selection is too large for one tool run. Select fewer sources.")
            documents, sources = {}, {identifier: [] for identifier in selected}
            document_keys = ("id", "name", "collection", "content_hash", "fingerprint",
                             "generation", "chunk_count")
            for row in rows:
                documents[row["id"]] = {key: row[key] for key in document_keys}
                source = {"id": row["chunk_id"], "document_id": row["id"],
                          "filename": row["name"], "locator": row["chunk_locator"],
                          "text": row["chunk_text"],
                          "source_path": str(self._managed_path(row["path"]))}
                if row["chunk_image_path"]:
                    source["image_path"] = str(self._managed_path(row["chunk_image_path"]))
                sources[row["id"]].append(source)
            if set(documents) != set(selected):
                raise ValueError("A selected source is unavailable, not indexed, or outside the active collection. "
                                 "Refresh the source list before generating.")
            if any(len(sources[identifier]) != documents[identifier]["chunk_count"]
                   for identifier in selected):
                raise ValueError("A source has an incomplete index. Reindex it before generating.")
            return {"documents": [documents[identifier] for identifier in selected],
                    "sources": [source for identifier in selected for source in sources[identifier]]}

    def validate_source_snapshot(self, documents):
        """Reject a changed/deleted source generation, including during a run."""
        with self._lock:
            for expected in documents:
                current = self._db.execute("SELECT * FROM documents WHERE id=?",
                                           (expected["id"],)).fetchone()
                if (current is None or current["status"] != "ready"
                        or current["fingerprint"] != self.settings.fingerprint()
                        or any(current[key] != expected[key] for key in
                               ("content_hash", "fingerprint", "generation", "collection", "chunk_count"))):
                    raise ValueError("A selected source changed during generation. Refresh sources and try again.")

    @staticmethod
    def _fts_query(query):
        # Quote every term so user input is never interpreted as FTS operators,
        # column selectors, or wildcards. Preserve compounds as phrase matches.
        words = re.findall(r"[^\W_]+(?:[_-][^\W_]+)*", query, flags=re.UNICODE)
        unique = list(dict.fromkeys(word.casefold() for word in words))[:64]
        return " OR ".join('"' + word.replace('"', '""') + '"' for word in unique)

    def search(self, query, query_vector, collection, document_ids=None, limit=8, candidates=20):
        from qdrant_client import models
        if not isinstance(query, str):
            raise ValueError("The search query must be text.")
        collection = self._label(collection, "Collection")
        if limit < 1 or candidates < 1:
            raise ValueError("Search limits must be positive.")
        candidates = max(limit, candidates)
        if isinstance(document_ids, str):
            raise ValueError("Selected documents must be a list of document identifiers.")
        selected = None if document_ids is None else set(document_ids)
        with self._lock:
            active = self._db.execute(
                "SELECT * FROM documents WHERE status='ready' AND collection=? AND fingerprint=? AND generation!=''",
                (collection, self.settings.fingerprint()),
            ).fetchall()
            active = [row for row in active if selected is None or row["id"] in selected]
            if not active:
                return []
            self._db.execute("DELETE FROM scoped_documents")
            self._db.executemany("INSERT INTO scoped_documents(id) VALUES (?)", [(row["id"],) for row in active])
            self._db.commit()
            rankings = []
            if query_vector is not None:
                vector = self._vector(query_vector, self.settings.embedding_dimensions)
                result = self.vector_client.query_points(
                    self._collection_name(self.settings.fingerprint(), self.settings.embedding_dimensions),
                    query=vector, query_filter=models.Filter(must=[
                        models.FieldCondition(key="collection", match=models.MatchValue(value=collection)),
                        models.FieldCondition(key="document_id", match=models.MatchAny(any=[row["id"] for row in active])),
                        models.FieldCondition(key="generation", match=models.MatchAny(any=[row["generation"] for row in active])),
                    ]), limit=candidates, with_payload=True, with_vectors=False,
                )
                rankings.append([point.payload["chunk_id"] for point in result.points
                                 if point.payload and "chunk_id" in point.payload])
            expression = self._fts_query(query)
            if expression:
                lexical = self._db.execute(
                    "SELECT c.id FROM chunks_fts JOIN chunks c ON c.rowid=chunks_fts.rowid "
                    "JOIN scoped_documents scope ON scope.id=c.document_id "
                    "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts), c.id LIMIT ?",
                    (expression, candidates),
                ).fetchall()
                rankings.append([row[0] for row in lexical])
            scores = {}
            for ranking in rankings:
                for rank, chunk_id in enumerate(dict.fromkeys(ranking), 1):
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (60 + rank)
            evidence = []
            for chunk_id in sorted(scores, key=lambda item: (-scores[item], item)):
                row = self._db.execute(
                    "SELECT c.*,d.name,d.path FROM chunks c JOIN documents d ON d.id=c.document_id "
                    "JOIN scoped_documents scope ON scope.id=d.id WHERE c.id=? AND d.status='ready' "
                    "AND d.fingerprint=? AND d.generation!=''", (chunk_id, self.settings.fingerprint()),
                ).fetchone()
                if row:
                    chunk = Chunk(**{key: row[key] for key in Chunk.__dataclass_fields__})
                    evidence.append(Evidence(chunk=chunk, filename=row["name"], score=scores[chunk_id],
                                             source_path=str(self._managed_path(row["path"]))))
                    if len(evidence) == limit:
                        break
            return evidence

    def close(self):
        with self._lock:
            self._db.close()
            if self._vectors is not None:
                self._vectors.close()
