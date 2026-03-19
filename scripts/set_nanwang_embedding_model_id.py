from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import lancedb


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    db_dir = os.environ.get("LANCEDB_DIR")
    if not db_dir:
        raise SystemExit("LANCEDB_DIR is not set")
    db_path = Path(db_dir).expanduser().resolve()
    print("LANCEDB_DIR =", str(db_path))
    if not db_path.exists():
        raise SystemExit("LANCEDB_DIR does not exist")

    # IMPORTANT: set to model hub ID so resolve_embedding_adapter can load it.
    target_model_id = "text-embedding-v4-openai-1"

    conn = lancedb.connect(str(db_path))
    meta = conn.open_table("collection_metadata")
    df = meta.search().where("name = '南网'").limit(10).to_pandas()
    if df is None or df.empty:
        raise SystemExit("collection_metadata 中找不到 '南网'")

    row: Dict[str, Any] = df.iloc[0].to_dict()
    print("old embedding_model_id =", row.get("embedding_model_id"))
    row["embedding_model_id"] = target_model_id
    row["updated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

    schema_names = list(meta.schema.names)
    cleaned = {k: _clean_value(row.get(k)) for k in schema_names}

    meta.delete("name = '南网'")
    meta.add([cleaned])

    df2 = meta.search().where("name = '南网'").limit(10).to_pandas()
    print("new embedding_model_id =", df2.iloc[0].get("embedding_model_id"))


if __name__ == "__main__":
    main()
