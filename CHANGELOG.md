# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **Documented `LANCEDB_AUTO_MIGRATE` in environment template**
  Added `LANCEDB_AUTO_MIGRATE` usage notes to `example.env`, including default behavior (`false`) and when to enable startup auto-migration.

### Changed

- **Knowledge Base upload: default parse method (breaking)**
  The default parse method on the KB detail upload form is now `"default"` instead of `"pypdf"`. The backend chooses the parser by file type (e.g. .docx, .pdf). If you rely on the previous default (always use PyPDF), select `"pypdf"` explicitly in the parse method dropdown when uploading.

- **Knowledge Base document registration (breaking)**
  Document IDs for new uploads are now generated deterministically from `(collection, source_path)` instead of a random UUID. Re-uploading the same file in the same collection updates the existing record instead of creating a duplicate.
  **Impact on existing data:** Documents that were registered with the previous logic (random UUID in `doc_id`) will get a *different* `doc_id` when re-uploaded. Re-uploading such a file will create a new record rather than updating the old one, so you may briefly see two entries for the same filename until the old one is removed. If you rely on idempotent re-uploads for previously registered documents, consider deleting the old document from the KB before re-uploading, or plan a one-time cleanup of legacy duplicates.

- **LanceDB user_id migration hardening**
  Startup and migration logic now include cross-process file locking, legacy `-1` orphan marker remapping to reserved int64 sentinel values, zero-progress loop protection, and shared embeddings-table listing utilities to avoid API-compat drift.
