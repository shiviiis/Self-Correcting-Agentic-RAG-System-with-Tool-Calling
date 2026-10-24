import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

"""Config 
 Absolute paths built from this file's location, so it works no matter
 where you launch Python or Streamlit from."""
 
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
PERSIST_DIR = str(BASE_DIR / "db" / "chroma_db")
MANIFEST_PATH = BASE_DIR / "db" / "manifest.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# If any of these change, old chunks are incompatible, so we rebuild everything.
CONFIG_SIGNATURE = f"{EMBEDDING_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}"


# Shared helpers (the retriever tool imports these too) 
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore():
    """Open (or create) the Chroma collection. Used by ingestion AND retrieval,
    so both are guaranteed to use the same path, embeddings, and distance metric."""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embedding_model(),
        collection_metadata={"hnsw:space": "cosine"},
    )


# Manifest: remembers which file versions are already embedded
def file_hash(path: Path) -> str:
    """SHA-256 of file contents. Content-based, so renaming/touching a file
    without editing it doesn't trigger re-embedding."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"config": CONFIG_SIGNATURE, "files": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# Loading and splitting 
def load_file(path: Path):
    """Load one file into LangChain Documents. PDFs yield one Document per page,
    with metadata['page'] (0-indexed), which you'll need for citations."""
    if path.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    # Normalize 'source' to just the filename. This is the key we use to delete
    # a file's chunks later, and what you'll show in citations.
    for d in docs:
        d.metadata["source"] = path.name
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # stores char offset in metadata, handy for debugging
    )
    return splitter.split_documents(documents)


#  Incremental ingestion 
def ingest(docs_dir: Path = DOCS_DIR) -> dict:
    """Sync the vector store with the docs folder. Safe to run repeatedly;
    only new or changed files are embedded. Returns a summary dict (handy for Streamlit)."""
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    vectorstore = get_vectorstore()

    # Rebuild from scratch if chunking/embedding settings changed
    if manifest.get("config") != CONFIG_SIGNATURE:
        print("Config changed, rebuilding the whole index")
        vectorstore.delete_collection()
        vectorstore = get_vectorstore()
        manifest = {"config": CONFIG_SIGNATURE, "files": {}}

    # Self-heal: manifest claims files are indexed but the DB is empty
    # (e.g., you deleted db/chroma_db but not manifest.json)
    if manifest["files"] and vectorstore._collection.count() == 0:
        print("Vector store is empty but manifest isn't, resetting manifest")
        manifest["files"] = {}

    # Diff the folder against the manifest
    current = {
        p.name: file_hash(p)
        for p in docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    old = manifest["files"]

    new_files = set(current) - set(old)
    removed_files = set(old) - set(current)
    changed_files = {n for n in current if n in old and old[n] != current[n]}

    # Delete stale chunks (for removed and changed files)
    for name in removed_files | changed_files:
        vectorstore._collection.delete(where={"source": name})
        print(f"Deleted old chunks for: {name}")

    # Embed new and changed files only
    chunks_added = 0
    for name in sorted(new_files | changed_files):
        docs = load_file(docs_dir / name)
        chunks = split_documents(docs)
        # Deterministic IDs: re-running can never create duplicates
        ids = [f"{name}:{current[name][:12]}:{i}" for i in range(len(chunks))]
        vectorstore.add_documents(chunks, ids=ids)
        chunks_added += len(chunks)
        print(f"Indexed {name}: {len(docs)} docs -> {len(chunks)} chunks")

    # Update manifest only after everything succeeded
    manifest["files"] = current
    save_manifest(manifest)

    summary = {
        "new": sorted(new_files),
        "changed": sorted(changed_files),
        "removed": sorted(removed_files),
        "chunks_added": chunks_added,
        "total_chunks": vectorstore._collection.count(),
    }
    print(f"Ingestion summary: {summary}")
    return summary


if __name__ == "__main__":
    ingest()