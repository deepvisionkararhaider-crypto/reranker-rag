// worker.js — 03-reranker-rag
// Two-stage retrieval: bi-encoder recall (Workers AI bge-small) -> cross-encoder
// rerank (Workers AI bge-reranker-base) -> top-k context -> Groq answer + citations.
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { chunkText, embed, cosine, rerank, chat, ensureSchema, uid } from './core.js';

const app = new Hono();
app.use('/api/*', cors());

const SAMPLE_DOCS = [
  {
    filename: 'reranking-explained.md',
    content: `Reranking is the second stage of a modern retrieval pipeline. Stage one, recall, uses a fast bi-encoder to fetch a broad candidate set of, say, the top fifty passages. Stage two, reranking, takes those candidates and scores each one against the query with a much more powerful model, then keeps only the best few for the language model.
The reason reranking works is a difference in model architecture. A bi-encoder embeds the query and each document independently, so it never lets the two interact; it is fast because document embeddings are precomputed. A cross-encoder instead feeds the query and the document together into a single transformer and outputs one relevance score. The two texts attend to each other token by token, which captures fine-grained relevance that a bi-encoder simply cannot see, but it is far too slow to run over an entire corpus. That is why cross-encoders are used only on a short candidate list.
A concrete example: for the query "how do I reset my password", a bi-encoder might retrieve a long generic security policy document because it is topically about passwords. A cross-encoder, seeing the exact question next to that document, will score it low and instead promote a short, specific "Password Reset Steps" article. This relevance ordering directly improves the answer the LLM produces.
Common rerankers include BGE reranker, Cohere Rerank, and cross-encoder models from sentence-transformers. Typical gains are large: reranked top-3 context often matches the answer quality of unrerked top-10 context, which means fewer tokens, lower latency downstream, and less distraction for the model.`,
  },
  {
    filename: 'bi-vs-cross-encoder.md',
    content: `A bi-encoder produces a fixed embedding for a piece of text. Queries and documents are encoded separately and compared by cosine similarity or dot product. The big advantage is precomputation: you embed your whole corpus once and store the vectors, so a query costs a single embedding plus a fast similarity scan. The limitation is that query and document never see each other during encoding.
A cross-encoder concatenates the query and the document and runs them jointly through a transformer, producing a single scalar relevance score. Because every query token attends to every document token, relevance judgments are far more accurate. The cost is that you cannot precompute anything: each query-document pair needs a fresh forward pass, so it scales with the number of candidates.
This is exactly why production systems combine them. Use the bi-encoder for cheap, scalable recall over millions of documents. Use the cross-encoder as a reranker over a few dozen candidates. The result is accuracy close to a cross-encoder with latency close to a bi-encoder.`,
  },
  {
    filename: 'two-stage-retrieval.md',
    content: `Two-stage retrieval is a standard pattern in search and RAG. Stage one optimizes for recall: cast a wide net and make sure the relevant passage is somewhere in the candidate set. Stage two optimizes for precision: reorder the candidates so the most relevant rise to the top.
A typical configuration retrieves fifty to one hundred candidates in stage one, then reranks and keeps the top three to five. Choosing the candidate count is a trade-off: too few and the reranker never sees the right passage, too many and latency climbs because the cross-encoder runs a forward pass per candidate.
Combining hybrid retrieval with reranking is the current best practice: fuse lexical BM25 and dense vectors for recall, then apply a cross-encoder for precision, then send a compact, high-quality context to the LLM. Each stage fixes the weakness of the previous one.`,
  },
];

async function seedIfEmpty(env) {
  const row = await env.DB.prepare('SELECT COUNT(*) AS n FROM documents').first();
  if (row && row.n > 0) return;
  for (const doc of SAMPLE_DOCS) await ingestDocument(env, doc.filename, doc.content);
}

async function ingestDocument(env, filename, content) {
  const docId = uid();
  const chunks = chunkText(content, 800, 120);
  if (!chunks.length) throw new Error('Document produced no chunks');
  const vectors = await embed(env, chunks);
  const now = Date.now();
  const stmts = [env.DB.prepare('INSERT INTO documents (id, filename, content, created_at) VALUES (?,?,?,?)').bind(docId, filename, content, now)];
  chunks.forEach((c, i) => {
    stmts.push(env.DB.prepare('INSERT INTO chunks (id, doc_id, filename, chunk_index, content, embedding, created_at) VALUES (?,?,?,?,?,?,?)')
      .bind(uid(), docId, filename, i, c, JSON.stringify(vectors[i]), now));
  });
  await env.DB.batch(stmts);
  return { docId, chunks: chunks.length };
}

async function recall(env, query, n) {
  const qv = (await embed(env, [query]))[0];
  const rows = (await env.DB.prepare('SELECT id, filename, chunk_index, content, embedding FROM chunks LIMIT 5000').all()).results || [];
  return rows.map((r) => ({ id: r.id, filename: r.filename, chunk_index: r.chunk_index, content: r.content, bi_score: cosine(qv, JSON.parse(r.embedding)) }))
    .sort((a, b) => b.bi_score - a.bi_score).slice(0, n);
}

app.get('/api/health', (c) => c.json({ status: 'healthy', service: 'reranker-rag', ts: Date.now() }));

app.get('/api/stats', async (c) => {
  await ensureSchema(c.env);
  const d = await c.env.DB.prepare('SELECT COUNT(*) AS n FROM documents').first();
  const ch = await c.env.DB.prepare('SELECT COUNT(*) AS n FROM chunks').first();
  return c.json({ documents: d?.n || 0, chunks: ch?.n || 0, reranker: '@cf/baai/bge-reranker-base', llm_model: c.env.LLM_MODEL || 'openai/gpt-oss-120b' });
});

app.get('/api/documents', async (c) => {
  await ensureSchema(c.env);
  try { await seedIfEmpty(c.env); } catch {}
  const rows = (await c.env.DB.prepare('SELECT d.id, d.filename, d.created_at, COUNT(ch.id) AS chunks FROM documents d LEFT JOIN chunks ch ON ch.doc_id=d.id GROUP BY d.id ORDER BY d.created_at DESC').all()).results || [];
  return c.json({ documents: rows, count: rows.length });
});

app.post('/api/documents', async (c) => {
  await ensureSchema(c.env);
  let body; try { body = await c.req.json(); } catch { return c.json({ error: 'Invalid JSON body' }, 400); }
  const filename = (body.filename || 'untitled.txt').toString().slice(0, 200);
  const text = (body.text || '').toString();
  if (!text.trim()) return c.json({ error: 'Field "text" is required' }, 400);
  try { const r = await ingestDocument(c.env, filename, text); return c.json({ document_id: r.docId, filename, chunk_count: r.chunks }, 201); }
  catch (e) { return c.json({ error: 'Ingestion failed', detail: String(e.message || e) }, 500); }
});

app.delete('/api/documents/:id', async (c) => {
  await ensureSchema(c.env);
  const id = c.req.param('id');
  await c.env.DB.batch([c.env.DB.prepare('DELETE FROM chunks WHERE doc_id=?').bind(id), c.env.DB.prepare('DELETE FROM documents WHERE id=?').bind(id)]);
  return c.json({ deleted: id });
});

// Search with rerank: reveal both stages for teaching
app.post('/api/search', async (c) => {
  await ensureSchema(c.env);
  let body; try { body = await c.req.json(); } catch { return c.json({ error: 'Invalid JSON body' }, 400); }
  const query = (body.query || '').toString();
  if (!query.trim()) return c.json({ error: 'Field "query" is required' }, 400);
  try { await seedIfEmpty(c.env); } catch {}
  const topK = Math.min(Math.max(parseInt(body.top_k) || 5, 1), 20);
  const candidates = Math.min(Math.max(parseInt(body.candidates) || 20, topK), 50);
  const t0 = Date.now();
  const recalled = await recall(c.env, query, candidates);
  const tRecall = Date.now();
  const scored = await rerank(c.env, query, recalled.map((r) => ({ id: r.id, text: r.content })));
  const scoreMap = Object.fromEntries(scored.map((s) => [s.id, s.score]));
  const reranked = recalled.map((r) => ({ ...r, rerank_score: Number((scoreMap[r.id] || 0).toFixed(6)) }))
    .sort((a, b) => b.rerank_score - a.rerank_score);
  // position before/after for teaching
  const before = recalled.map((r) => r.id);
  const after = reranked.map((r) => r.id);
  const results = reranked.slice(0, topK).map((r) => ({ ...r, rank_before: before.indexOf(r.id) + 1, rank_after: after.indexOf(r.id) + 1 }));
  return c.json({ query, mode: 'recall+rerank', results, candidates_considered: recalled.length, recall_time_ms: tRecall - t0, rerank_time_ms: Date.now() - tRecall, total_time_ms: Date.now() - t0 });
});

app.post('/api/query', async (c) => {
  await ensureSchema(c.env);
  let body; try { body = await c.req.json(); } catch { return c.json({ error: 'Invalid JSON body' }, 400); }
  const query = (body.query || '').toString();
  if (!query.trim()) return c.json({ error: 'Field "query" is required' }, 400);
  try { await seedIfEmpty(c.env); } catch {}
  const topK = Math.min(Math.max(parseInt(body.top_k) || 4, 1), 12);
  const candidates = Math.min(Math.max(parseInt(body.candidates) || 20, topK), 50);
  const t0 = Date.now();
  const recalled = await recall(c.env, query, candidates);
  const tR1 = Date.now();
  const scored = await rerank(c.env, query, recalled.map((r) => ({ id: r.id, text: r.content })));
  const scoreMap = Object.fromEntries(scored.map((s) => [s.id, s.score]));
  const reranked = recalled.map((r) => ({ ...r, rerank_score: scoreMap[r.id] || 0 })).sort((a, b) => b.rerank_score - a.rerank_score).slice(0, topK);
  const tR2 = Date.now();
  if (!reranked.length) return c.json({ query, answer: "I don't have enough information in the provided documents to answer this question.", citations: [], model: 'none', recall_time_ms: tR1 - t0, rerank_time_ms: tR2 - tR1, generation_time_ms: 0, total_time_ms: Date.now() - t0 });
  const context = reranked.map((ch, i) => `[Source ${i + 1}] ${ch.filename} (chunk ${ch.chunk_index})\n${ch.content}`).join('\n\n---\n\n');
  const answer = await chat(c.env, [
    { role: 'system', content: 'You are a precise RAG assistant. Context was reranked by a cross-encoder for relevance. Answer ONLY from context. Cite [Source N]. Never invent facts.' },
    { role: 'user', content: `Context:\n${context}\n\nQuestion: ${query}\n\nAnswer with citations:` },
  ]);
  const tG = Date.now();
  return c.json({
    query, answer,
    citations: reranked.map((ch, i) => ({ n: i + 1, chunk_id: ch.id, filename: ch.filename, chunk_index: ch.chunk_index, bi_score: Number(ch.bi_score.toFixed(4)), rerank_score: Number(ch.rerank_score.toFixed(4)), content: ch.content.slice(0, 300) })),
    model: c.env.LLM_MODEL || 'openai/gpt-oss-120b',
    recall_time_ms: tR1 - t0, rerank_time_ms: tR2 - tR1, generation_time_ms: tG - tR2, total_time_ms: tG - t0,
  });
});

app.get('/', (c) => c.html(UI));

const UI = `<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reranker RAG — bi-encoder + cross-encoder</title>
<script src="https://cdn.tailwindcss.com"></script><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css">
<style>body{background:#0b1020}.glass{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08)}</style></head>
<body class="min-h-screen text-slate-100">
<header class="border-b border-white/10"><div class="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
<div class="flex items-center gap-3"><div class="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-rose-500 grid place-items-center"><i class="fas fa-layer-group"></i></div>
<div><h1 class="font-bold text-lg leading-tight">Reranker RAG</h1><p class="text-xs text-slate-400">Bi-encoder recall → cross-encoder rerank → LLM</p></div></div>
<div id="stats" class="text-xs text-slate-400"></div></div></header>
<main class="max-w-6xl mx-auto px-4 py-6 grid lg:grid-cols-2 gap-6">
<section class="glass rounded-2xl p-5"><h2 class="font-semibold mb-3"><i class="fas fa-file-arrow-up mr-2 text-amber-400"></i>Knowledge base</h2>
<input id="fname" placeholder="filename.txt" class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm mb-2"/>
<textarea id="dtext" rows="4" placeholder="Paste text…" class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm mb-2"></textarea>
<div class="flex flex-wrap gap-2 mb-4"><button onclick="ingest()" class="bg-amber-600 hover:bg-amber-500 rounded-lg px-4 py-2 text-sm"><i class="fas fa-plus mr-1"></i>Ingest</button>
<label class="cursor-pointer bg-white/10 hover:bg-white/20 rounded-lg px-3 py-2 text-sm"><i class="fas fa-paperclip mr-1"></i>Upload<input type="file" accept=".txt,.md,.markdown" class="hidden" onchange="loadFile(event)"></label>
<button onclick="seed()" class="bg-white/5 hover:bg-white/10 rounded-lg px-3 py-2 text-sm">Load samples</button></div>
<h3 class="text-xs uppercase tracking-wide text-slate-400 mb-2">Documents (<span id="dcount">0</span>)</h3>
<ul id="docs" class="space-y-1 max-h-52 overflow-auto text-sm"></ul></section>
<section class="glass rounded-2xl p-5"><h2 class="font-semibold mb-3"><i class="fas fa-wand-magic-sparkles mr-2 text-rose-400"></i>Two-stage query</h2>
<div class="flex gap-2 mb-2"><input id="q" placeholder="e.g. how do bi-encoders differ from cross-encoders?" class="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm" onkeydown="if(event.key==='Enter')ask()"/>
<button onclick="ask()" class="bg-rose-600 hover:bg-rose-500 rounded-lg px-4 py-2 text-sm"><i class="fas fa-paper-plane"></i></button></div>
<div class="flex items-center gap-3 text-xs text-slate-400 mb-3"><span>Candidates</span><input id="cand" type="range" min="5" max="40" step="5" value="20" class="flex-1" oninput="document.getElementById('cv').textContent=this.value"><span id="cv" class="text-rose-300">20</span></div>
<div id="meta" class="text-xs text-slate-400 mb-2"></div>
<div id="answer" class="text-sm leading-relaxed whitespace-pre-wrap min-h-[80px]"></div>
<div id="cites" class="mt-4 space-y-2"></div></section></main>
<p class="text-center text-xs text-slate-500 py-6">Workers AI bge-small (recall) + bge-reranker-base (cross-encoder) · Groq · D1 · <span id="health">…</span></p>
<script>
const api=(p,o)=>fetch(p,o).then(r=>r.json());
async function refresh(){try{const s=await api('/api/stats');document.getElementById('stats').innerHTML='<i class="fas fa-database mr-1"></i>'+s.documents+' docs · '+s.chunks+' chunks · reranker '+s.reranker;
const d=await api('/api/documents');document.getElementById('dcount').textContent=d.count;
document.getElementById('docs').innerHTML=d.documents.map(x=>'<li class="flex items-center justify-between bg-white/5 rounded-lg px-3 py-2"><span class="truncate"><i class="fas fa-file-lines mr-2 text-slate-400"></i>'+x.filename+' <span class="text-slate-500">('+x.chunks+')</span></span><button onclick="del(\\''+x.id+'\\')" class="text-slate-400 hover:text-rose-400"><i class="fas fa-trash"></i></button></li>').join('')||'<li class="text-slate-500 text-xs px-3 py-2">No documents.</li>';}catch(e){}}
async function ingest(){const filename=document.getElementById('fname').value||('doc-'+Date.now()+'.txt');const text=document.getElementById('dtext').value;if(!text.trim()){alert('Enter text');return;}
const r=await fetch('/api/documents',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename,text})});const j=await r.json();if(j.error){alert(j.error);return;}document.getElementById('dtext').value='';document.getElementById('fname').value='';refresh();}
function loadFile(e){const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{document.getElementById('dtext').value=r.result;document.getElementById('fname').value=f.name;};r.readAsText(f);}
async function del(id){await fetch('/api/documents/'+id,{method:'DELETE'});refresh();}
async function ask(){const q=document.getElementById('q').value.trim();if(!q)return;const candidates=parseInt(document.getElementById('cand').value);
document.getElementById('answer').innerHTML='<i class="fas fa-spinner fa-spin text-rose-400"></i>';document.getElementById('cites').innerHTML='';document.getElementById('meta').textContent='';
try{const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,candidates})});const j=await r.json();if(j.error){document.getElementById('answer').textContent='Error: '+j.error;return;}
document.getElementById('answer').textContent=j.answer;
document.getElementById('meta').innerHTML='<i class="fas fa-bolt mr-1 text-amber-400"></i>'+j.model+' · recall '+j.recall_time_ms+'ms · rerank '+j.rerank_time_ms+'ms · generate '+j.generation_time_ms+'ms';
document.getElementById('cites').innerHTML=(j.citations||[]).map(c=>'<details class="bg-white/5 rounded-lg px-3 py-2 text-sm"><summary class="cursor-pointer"><span class="text-amber-300 font-medium">[Source '+c.n+'] '+c.filename+'</span> <span class="text-slate-500">· bi '+c.bi_score+' → rerank '+c.rerank_score+'</span></summary><p class="mt-2 text-slate-300 text-xs">'+c.content.replace(/</g,'&lt;')+'…</p></details>').join('');
}catch(e){document.getElementById('answer').textContent='Failed: '+e;}}
async function seed(){document.getElementById('answer').innerHTML='<span class="text-slate-400 text-xs">Seeding…</span>';await api('/api/documents');refresh();document.getElementById('answer').innerHTML='';}
api('/api/health').then(h=>document.getElementById('health').innerHTML='<span class="text-emerald-400">API '+h.status+'</span>');refresh();
</script></body></html>`;

export default app;
