#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-http://localhost:8787}"
pass=0; fail=0
check(){ if [ "$1" = "$2" ]; then echo "PASS: $3"; pass=$((pass+1)); else echo "FAIL: $3 (got '$1' want '$2')"; fail=$((fail+1)); fi; }
check "$(curl -s "$BASE/api/health" | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")" "healthy" "health"
check "$(curl -s "$BASE/api/stats" | python3 -c "import sys,json;print(json.load(sys.stdin)['reranker'])")" "@cf/baai/bge-reranker-base" "stats reranker"
code=$(curl -s -o /tmp/r3.json -w "%{http_code}" -X POST "$BASE/api/documents" -H 'Content-Type: application/json' -d '{"filename":"t.txt","text":"A cross-encoder reranker scores query and document jointly. A bi-encoder encodes them independently."}')
check "$code" "201" "ingest"
# rerank search returns rerank_score and rank_before/after
out=$(curl -s -X POST "$BASE/api/search" -H 'Content-Type: application/json' -d '{"query":"cross-encoder rerank","candidates":10}')
has=$(echo "$out" | python3 -c "import sys,json;d=json.load(sys.stdin);print('ok' if d['results'] and 'rerank_score' in d['results'][0] and 'rank_before' in d['results'][0] else 'bad')")
check "$has" "ok" "rerank returns cross-encoder scores + rank shifts"
check "$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/api/search" -H 'Content-Type: application/json' -d '{"query":""}')" "400" "empty query rejected"
if curl -s -X POST "$BASE/api/query" -H 'Content-Type: application/json' -d '{"query":"why use a reranker?"}' | grep -q '"answer"'; then echo "PASS: query returns answer"; pass=$((pass+1)); else echo "FAIL: query"; fail=$((fail+1)); fi
echo "---- $pass passed, $fail failed ----"; [ "$fail" -eq 0 ]
