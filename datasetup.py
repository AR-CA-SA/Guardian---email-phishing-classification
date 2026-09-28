from pathlib import Path
import numpy as np
import pandas as pd


class processor(Path):
    def __init__(self):
        pass
    def loadDataNPZ(importPath:Path)->np.lib.npyio.NpzFile:
        DATA_PATH = Path(importPath)

        index  = np.load(DATA_PATH, allow_pickle=False )
        n_rows = int(index["_rows"])
        n_features =  int(index["_features"])

        print(f"NPZ index loaded:")
        print(f"  Samples:  {n_rows:,}")
        print(f"  Features: {n_features:,}")
        print(f"  Referenced files: X={index['_X_file']}, y={index['_y_file']}")
        print(f" Type : {type(index)}")
        return index
    def loadDataFM(importPath:Path,index: np.lib.npyio.NpzFile)->np.float32:

        n_rows = int(index["_rows"])
        n_features =  int(index["_features"])
        
        X = np.load(importPath, mmap_mode="r")
        assert X.shape == (n_rows, n_features), f"Shape mistmatch : {X.shape} vs ({n_rows}, {n_features})" 
        print(f" Feature matrix opened: {X.shape[0]:,} samples x {X.shape[1]:,} features")
        print(f" dtype: {X.dtype} — memory-mapped (~37 MB, fits entirely in RAM)")
        return X
    def loadMetaData(importPath:str, X:np.float32, index: np.lib.npyio.NpzFile)->tuple[np.int32, pd.core.frame.DataFrame]:
        n_rows = int(index["_rows"])
        n_features =  int(index["_features"])
        
        meta = pd.read_parquet(importPath)
        y = meta["label_int"].values

        assert X.shape[0] == len(y), f"Row mismatch: X={X.shape[0]}, y={len(y)}"
        assert set(np.unique(y)) == {0, 1}, f"Unexpected labels: {np.unique(y)}"
    
        print(f"Metadata loaded: {meta.shape[0]:,} rows × {meta.shape[1]} columns")
        print(f"Columns: {list(meta.columns)}")
        print()
        print("Column notes:")
        print("  - 'timestamp'               : null for all rows — not native to email datasets.")
        print("    is part of the canonical schema used across all IT4LIA datasets")
        print("    for future merges and cross-dataset comparisons.")
        print("  - 'email_*'                 : dataset-specific metadata (sender, receiver, date,")
        print("    subject, source_file, urls_flag, has_header, id_source).")
        print("  - 'flag_timestamp_missing'  : True for all rows (no structured timestamp in email datasets).")
        print("  - 'hash_type'               : constant 'sha256'.")
        print("  - '_group_key', '_keep_row', '_dedup_reason', etc. : dedup audit trail from Step 2.1B.")
        print()
        print(f"{type(meta)}")
        print(f"{y.dtype}")
        print(f"Labels:  {(y == 1).sum():,} phishing  |  {(y == 0).sum():,} legitimate")
        print(f"\n✅ All checks passed: {n_rows:,} samples × {n_features:,} features, labels ∈ {{0, 1}}")
        print(f"labels : 0 Legitimate — genuine, safe email, 1 Phishing — email designed to deceive users")
        return y, meta
