# Fixing Undefined Names (F821) in Code Examples

## Problem

The code examples are described as "production-ready" but contain 1,444 F821 errors (undefined names), making them non-runnable.

## Analysis

Three categories of F821 errors:

### 1. Missing Imports
- `np` (numpy)
- `faiss`
- `torch`
- `cosine_similarity`
**Fix**: Add proper imports

### 2. Undefined Example Data
- `documents`, `query`, `sentences` (in example snippets)
- `dim`, `embeddings` (in demonstrations)
**Fix**: Add example data at top of file

### 3. Abstract Dependencies
- `encoder` (represents any embedding model)
- `llm` (represents any language model)
- `train_from_scratch` (represents training function)
**Fix**: Add comments explaining OR add placeholder implementations

## Recommendation

Given 1,444 errors across 253 files, we have options:

### Option A: Fix All (Comprehensive but time-consuming)
- Make every file truly runnable
- Add all imports, example data, placeholders
- Time: 4-6 hours
- Result: 0 F821 errors

### Option B: Fix Representative Sample (Pragmatic)
- Fix Ch01 examples (16 files) to establish pattern
- Document the pattern in LINTING.md
- Add note that other chapters follow same approach  
- Time: 30-45 minutes
- Result: ~100 F821 errors fixed, pattern established

### Option C: Update Documentation (Honest but incomplete)
- Change "production-ready" to "pedagogical examples"
- Add note that examples may require setup
- Keep code focused on concepts
- Time: 10 minutes
- Result: 1,444 F821 errors remain but accurately described

### Option D: Hybrid Approach (Balanced)
- Fix simple cases (missing imports): ~200 errors
- Add example data to snippet files: ~100 errors  
- Add comments for abstract deps: ~100 errors
- Update docs for remaining: ~1,044 errors
- Time: 1-2 hours
- Result: 60% reduction, improved usability

## User's Point

The user is right: claiming "production-ready" with 1,444 undefined names is misleading. We should either:
1. Fix them to BE production-ready, or
2. Accurately describe what they are

What would you like me to do?
