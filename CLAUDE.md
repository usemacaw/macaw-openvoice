# Macaw OpenVoice

Runtime unificado de voz (STT + TTS) construido do zero em Python. Orquestra engines de inferencia (Faster-Whisper, Silero VAD, Kokoro, Qwen3-TTS) com API OpenAI-compatible.

**Status: M9 (Full-Duplex) completo. Todas as fases do PRD entregues.** M1-M9 completos. 1700+ testes passando. Runtime unificado STT+TTS funcional.

## Commands

```bash
# Development (always use make targets — they use .venv/bin/ automatically)
make check              # format + lint + typecheck
make test               # all tests
make test-unit          # unit tests only (prefer during development)
make test-integration   # integration tests only
make test-fast          # all tests except @pytest.mark.slow
make ci                 # full pipeline: format + lint + typecheck + test
make proto              # generate protobuf stubs

# Individual test
.venv/bin/python -m pytest tests/unit/test_foo.py::test_bar -q
```

## Architecture

```
macaw/
├── server/           # FastAPI — endpoints REST + WebSocket
│   ├── routes/       # transcriptions, translations, speech, voices, health, realtime
│   └── voice_store.py # Voice persistence (VoiceStore ABC + FileSystemVoiceStore)
├── scheduler/        # Async priority queue, cancellation, batching, latency tracking, TTS converters, metrics
├── registry/         # Model Registry (macaw.yaml, lifecycle)
├── workers/          # Subprocess gRPC management
│   ├── stt/          # STTBackend interface + FasterWhisperBackend
│   └── tts/          # TTSBackend interface + KokoroBackend + Qwen3TTSBackend
├── preprocessing/    # Audio pipeline (resample, DC remove, gain normalize)
├── postprocessing/   # Text pipeline (ITN via NeMo, fail-open)
├── vad/              # Voice Activity Detection (energy pre-filter + Silero)
├── session/          # Session Manager (state machine, ring buffer, WAL, recovery, backpressure, mute, metrics)
├── cli/              # CLI commands (click)
└── proto/            # gRPC protobuf definitions (STT + TTS)
```

Full details: @docs/ARCHITECTURE.md
Roadmap: @docs/ROADMAP.md
PRD: @docs/PRD.md

## Key Design Decisions

- **Um binario, um processo, dois tipos de worker.** STT e TTS compartilham API Server, Registry, Scheduler, CLI. Workers sao subprocessos gRPC isolados.
- **Model-agnostic via campo `architecture`** no manifesto: `encoder-decoder` (Whisper), `ctc`, `streaming-native` (Paraformer). O runtime adapta o pipeline automaticamente.
- **VAD no runtime, nao na engine.** Silero VAD como biblioteca, com energy pre-filter proprio. Garante comportamento consistente entre engines.
- **Preprocessing e post-processing sao responsabilidade do runtime**, nao da engine. Engines recebem PCM 16kHz normalizado e retornam texto cru.
- **Workers sao subprocessos gRPC** — crash do worker nao derruba o runtime. Recovery via WAL in-memory.

- **Pipeline adaptativo por `architecture`**: `StreamingSession` adapta comportamento automaticamente — encoder-decoder usa LocalAgreement + cross-segment context; CTC usa partials nativos, sem LocalAgreement, sem cross-segment context.
- **Full-duplex via mute-on-speak.** STT e TTS na mesma conexao WebSocket. Quando TTS esta ativo, STT descarta frames (mute). Unmute garantido via try/finally.
- **TTS e stateless por request.** Nao reusar Session Manager para TTS. Cada `tts.speak` e independente.
- **TTSBackend espelha STTBackend.** `synthesize()` retorna `AsyncIterator[bytes]` para streaming com baixo TTFB.

ADRs completos: @docs/PRD.md (secao "Architecture Decision Records") + @docs/adrs/
Como adicionar nova engine: @docs/ADDING_ENGINE.md

## Code Style

- Python 3.12 (via `uv`), tipagem estrita com mypy
- Async-first: todas as interfaces publicas sao `async`
- Formatacao: ruff (format + lint)
- Imports: absolutos a partir de `macaw.` (ex: `from macaw.registry import Registry`)
- Nomenclatura: snake_case para funcoes/variaveis, PascalCase para classes
- Docstrings: apenas em interfaces publicas (ABC) e funcoes nao-obvias
- Sem comentarios obvios — o codigo deve ser auto-explicativo
- Erros: exceptions tipadas por dominio, nunca `Exception` generico
- Commits seguem conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`

## Testing

- Framework: pytest + pytest-asyncio com `asyncio_mode = "auto"` (sem `@pytest.mark.asyncio`)
- Async FastAPI tests: `httpx.AsyncClient` com `ASGITransport`
- Para testar error handlers (500): `ASGITransport(raise_app_exceptions=False)`
- Fixtures em `tests/conftest.py` (audio WAV sine tones 440Hz gerados automaticamente)
- Mocks: usar `unittest.mock` para engines de inferencia nos testes unitarios
- Testes de integracao marcados com `@pytest.mark.integration`
- **IMPORTANTE**: Sempre rodar `make test-unit` durante desenvolvimento, nao a suite inteira

## Things That Will Bite You

- **gRPC streams sao o heartbeat.** Nao implemente health check polling separado para detectar crash de worker — o stream break do gRPC bidirecional e a detecao.
- **Ring buffer tem read fence.** Nunca sobrescrever dados apos `last_committed_offset` — sao necessarios para recovery.
- **ITN so em `transcript.final`.** Nunca aplicar ITN em `transcript.partial` — partials sao instaveis e ITN geraria output confuso.
- **Preprocessing vem ANTES do VAD.** O audio deve estar normalizado antes de chegar no Silero VAD, senao os thresholds nao funcionam.
- **`vad_filter: false` no manifesto.** O VAD e do runtime. Nao habilitar o VAD interno da engine (ex: Faster-Whisper) — duplicaria o trabalho.
- **Session Manager e so STT.** TTS e stateless por request. Nao tentar reusar Session Manager para TTS.
- **LocalAgreement e para encoder-decoder apenas.** CTC e streaming-native tem partials nativos — nao aplicar LocalAgreement neles. O `StreamingSession` faz isso automaticamente via `_architecture` field.
- **CTC nao usa cross-segment context.** CTC nao suporta `initial_prompt` — `_build_initial_prompt()` retorna `None` para CTC (exceto hot words sem suporte nativo).
- **Manifest `type` field** e normalizado para `model_type` no `ModelManifest` (evita conflito com built-in Python).
- **Force commit e assincrono.** O RingBuffer callback `on_force_commit` e sincrono (chamado de `write()`), mas seta um flag `_force_commit_pending` que `process_frame()` (async) verifica. Nunca bloquear a escrita de novos frames dentro do callback.
- **SessionStateMachine e frozen.** Transicoes invalidas levantam `InvalidTransitionError`. Estado CLOSED e terminal — nenhuma transicao permitida depois.
- **WAL checkpoint e monotonic.** Usa `time.monotonic()` para timestamps, nunca `time.time()`. Garante consistencia mesmo com ajustes de relogio.
- **Recovery reenvia uncommitted data.** Apos crash do worker, `recover()` reenvia `ring_buffer.get_uncommitted()` ao novo stream. Nao duplica dados ja commitados gracas ao read fence.
- **Streaming NAO passa pela PriorityQueue.** WebSocket streaming usa `StreamingGRPCClient` diretamente. A PriorityQueue do Scheduler e apenas para requests batch (REST API).
- **Scheduler.transcribe() e backward-compatible.** A assinatura externa de M3 e mantida. Internamente usa `_transcribe_inline()` que faz submit+await do future.
- **CancellationManager remove entry no cancel.** Apos `cancel()`, a request e removida do tracking. `unregister()` e no-op se ja foi cancelada.
- **BatchAccumulator flush e fire-and-forget.** O flush callback (`_dispatch_batch`) e chamado pelo timer asyncio. Se o scheduler para antes do flush, `stop()` faz flush manual.
- **LatencyTracker usa TTL.** Entries expiram apos 300s. `cleanup()` deve ser chamado periodicamente para evitar memory leak em requests que nunca completam.
- **Metricas do Scheduler sao opcionais.** Usam `try/except ImportError` identico a `macaw.session.metrics`. Sempre verificar `if metric is not None` antes de observar.
- **Mute-on-speak e try/finally.** `_tts_speak_task()` chama `session.mute()` antes do TTS e `session.unmute()` em `finally`. Se o TTS crashar, o unmute AINDA acontece. Nunca mutar sem garantir unmute.
- **TTS binary frames sao server->client.** No WebSocket, binary frames server->client sao SEMPRE audio TTS. Client->server sao SEMPRE audio STT. Sem ambiguidade de direcao.
- **`tts.speak` cancela o anterior.** Se um `tts.speak` chegar enquanto outro esta em andamento, o anterior e cancelado automaticamente (cancel_event.set()). Nao acumula.
- **TTS worker e subprocess separado.** O TTS worker roda em porta diferente (default 50052 vs 50051 para STT). A factory `_create_backend("kokoro")` faz lazy import da engine.
- **Metricas TTS sao opcionais.** Usam mesmo padrao lazy import (`try/except ImportError` + `HAS_TTS_METRICS` flag) que `macaw.session.metrics`. Sempre verificar antes de observar.
- **`POST /v1/audio/speech` retorna WAV ou PCM.** Default e WAV com header. `response_format=pcm` retorna raw PCM 16-bit.
- **VoiceStore e opcional.** `app.state.voice_store` pode ser `None`. Rotas de CRUD de vozes retornam 400 se nao configurado. Dependency `get_voice_store()` retorna `None` nesse caso.
- **Saved voice usa prefixo `voice_`.** Na rota speech, `voice` comecando com `voice_` e resolvido via VoiceStore. O prefixo e removido para lookup. Apos resolucao, `voice` vira `"default"` para a engine.
- **Conflito ref_audio inline + saved voice.** Se o request traz `ref_audio` inline E o saved voice tem ref_audio, retorna 400 (InvalidRequestError). Nao sobreescreve silenciosamente.

## Workflow

- Ler o PRD (@docs/PRD.md) antes de implementar qualquer componente
- Consultar a arquitetura (@docs/ARCHITECTURE.md) para entender onde cada peca se encaixa
- Ao adicionar nova engine STT: seguir guia em @docs/ADDING_ENGINE.md (5 passos: implementar `STTBackend` ABC, criar manifesto `macaw.yaml`, registrar na factory, declarar dependencia, escrever testes). Zero mudancas no runtime core.
- Ao adicionar novo stage de preprocessing/postprocessing: seguir o padrao pipeline existente (cada stage e toggleavel via config)
- Full-duplex guide: @docs/FULL_DUPLEX.md para integracao STT+TTS no mesmo WebSocket

## Environment

- **Python 3.12** via `uv` (sistema tem 3.10, projeto requer >=3.11)
- Venv: `.venv/` criado com `uv venv --python 3.12`
- Todos os comandos devem usar `.venv/bin/` ou `make` targets (que ja usam `.venv/bin/`)
- CUDA opcional (fallback para CPU transparente)
- Dependencias pesadas (Faster-Whisper, nemo_text_processing) sao opcionais por engine
- gRPC tools necessarios para gerar protobufs: `grpcio-tools`

## API Contracts

- REST: compativel com OpenAI Audio API (`/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/audio/speech`)
- REST: Voice management (`GET /v1/voices` lista vozes preset dos workers, `POST /v1/voices` cria saved voice, `GET /v1/voices/{id}`, `DELETE /v1/voices/{id}`)
- WebSocket: protocolo de eventos JSON original (`/v1/realtime`) — STT streaming + TTS full-duplex. Comandos: `tts.speak`, `tts.cancel`. Eventos: `tts.speaking_start`, `tts.speaking_end`.
- gRPC: protocolo interno runtime <-> worker (nao exposto a clientes). STT: `stt_worker.proto`. TTS: `tts_worker.proto` (inclui `ListVoices` RPC).

Contratos detalhados: @docs/PRD.md (secoes 9-13)