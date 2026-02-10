'use client';

import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import useSWR, { mutate } from 'swr';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select';
import { Slider } from '../components/ui/slider';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import {
  cancelJob,
  createJob,
  createRealtimeConnection,
  fetchJobs,
  fetchModels,
  fetchQueue,
  sendTtsCancel,
  sendTtsSpeak,
  synthesizeSpeech,
  type DemoJob,
  type DemoModel,
  type QueueMetrics,
} from '../lib/api';

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const PRIORITY_OPTIONS = [
  { value: 'REALTIME', label: 'Realtime' },
  { value: 'BATCH', label: 'Batch' },
];

function formatDate(timestamp: number) {
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'short',
    timeStyle: 'medium',
  }).format(new Date(timestamp * 1000));
}

function statusVariant(status: string) {
  switch (status) {
    case 'completed':
      return 'success';
    case 'failed':
      return 'destructive';
    case 'cancelled':
      return 'secondary';
    default:
      return 'outline';
  }
}

// ---------------------------------------------------------------------------
// Icons (inline SVG to avoid adding lucide-react dependency issues)
// ---------------------------------------------------------------------------

function MicIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" x2="12" y1="19" y2="22" />
    </svg>
  );
}

function StopIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className}>
      <rect x="6" y="6" width="12" height="12" rx="1" />
    </svg>
  );
}

function VolumeIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
    </svg>
  );
}

function VolumeOffIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <line x1="22" x2="16" y1="9" y2="15" />
      <line x1="16" x2="22" y1="9" y2="15" />
    </svg>
  );
}

function SendIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <line x1="22" x2="11" y1="2" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Hero Section
// ---------------------------------------------------------------------------

function HeroSection({ modelCount }: { modelCount: number }) {
  return (
    <section className="relative overflow-hidden rounded-2xl border bg-card px-8 py-12">
      {/* Background decoration */}
      <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-primary/5 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-20 -left-20 h-64 w-64 rounded-full bg-primary/5 blur-3xl" />

      <div className="relative z-10 flex flex-col items-center text-center">
        <div className="mb-4 flex items-center gap-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <VolumeIcon className="h-5 w-5" />
          </div>
          <span className="text-sm font-medium tracking-wider text-muted-foreground uppercase">
            Macaw OpenVoice
          </span>
        </div>

        <h1 className="gradient-text text-4xl font-bold tracking-tight sm:text-5xl">
          Voice Runtime
        </h1>

        <p className="mt-4 max-w-2xl text-lg text-muted-foreground">
          STT + TTS in a single server. OpenAI-compatible API, WebSocket streaming,
          smart VAD, session management, and failure recovery.
        </p>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Badge variant="outline" className="px-3 py-1.5 text-sm">
            {modelCount} model{modelCount !== 1 ? 's' : ''} installed
          </Badge>
          <Badge variant="outline" className="px-3 py-1.5 text-sm">
            OpenAI-compatible API
          </Badge>
          <Badge variant="outline" className="px-3 py-1.5 text-sm">
            Full-Duplex STT + TTS
          </Badge>
          <Badge variant="outline" className="px-3 py-1.5 text-sm">
            gRPC Worker Isolation
          </Badge>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Dashboard Tab
// ---------------------------------------------------------------------------

interface DashboardTabProps {
  models: DemoModel[] | undefined;
  queue: QueueMetrics | undefined;
  jobs: DemoJob[] | undefined;
  defaultModel: string;
}

function DashboardTab({ models, queue, jobs, defaultModel }: DashboardTabProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [priority, setPriority] = useState<string>('REALTIME');
  const [language, setLanguage] = useState<string>('');
  const [submitState, setSubmitState] = useState<{
    isSubmitting: boolean;
    error?: string;
    success?: string;
  }>({ isSubmitting: false });

  const effectiveModel = selectedModel || defaultModel;

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setSelectedFile(file ?? null);
  }

  async function onSubmit() {
    if (!selectedFile) {
      setSubmitState({ isSubmitting: false, error: 'Select an audio file.' });
      return;
    }
    if (!effectiveModel) {
      setSubmitState({ isSubmitting: false, error: 'No models available.' });
      return;
    }

    setSubmitState({ isSubmitting: true });
    try {
      await createJob({
        file: selectedFile,
        model: effectiveModel,
        priority,
        language: language || undefined,
      });
      setSubmitState({ isSubmitting: false, success: 'Job submitted successfully.' });
      setSelectedFile(null);
      mutate('jobs');
      mutate('queue');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setSubmitState({ isSubmitting: false, error: message });
    }
  }

  async function onCancel(requestId: string) {
    try {
      await cancelJob(requestId);
      mutate('jobs');
      mutate('queue');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to cancel';
      setSubmitState((prev) => ({ ...prev, error: message }));
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2">
        {/* Models card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded bg-primary/10 text-xs font-bold text-primary">M</span>
              Installed Models
            </CardTitle>
            <CardDescription>
              {models?.length ? `${models.length} model(s) available` : 'Loading models...'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {models === undefined ? (
              // Loading skeleton
              Array.from({ length: 2 }).map((_, i) => (
                <div key={i} className="animate-pulse rounded-lg border p-3">
                  <div className="flex items-center justify-between">
                    <div className="h-4 w-40 rounded bg-muted" />
                    <div className="h-5 w-16 rounded-full bg-muted" />
                  </div>
                  <div className="mt-2 h-3 w-56 rounded bg-muted" />
                  <div className="mt-2 flex gap-2">
                    <div className="h-5 w-12 rounded-full bg-muted" />
                    <div className="h-5 w-10 rounded-full bg-muted" />
                  </div>
                </div>
              ))
            ) : models.length > 0 ? (
              models.map((model) => (
                <div key={model.name} className="fade-slide-in rounded-lg border p-3 transition-colors hover:bg-muted/50">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold">{model.name}</p>
                    <Badge variant="outline">{model.engine}</Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">{model.description}</p>
                  <div className="mt-1.5 flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">v{model.version}</Badge>
                    <Badge variant="secondary" className="text-xs">{model.type.toUpperCase()}</Badge>
                    {model.capabilities?.architecture && (
                      <Badge variant="secondary" className="text-xs">{String(model.capabilities.architecture)}</Badge>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-lg border border-dashed p-4 text-center">
                <p className="text-sm font-medium">No models installed</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Run <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">macaw pull faster-whisper-tiny</code> to get started.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Queue card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded bg-primary/10 text-xs font-bold text-primary">Q</span>
              Scheduler Queue
            </CardTitle>
            <CardDescription>Queue depth and real-time priorities.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between rounded-lg bg-muted/50 p-4">
              <span className="text-sm text-muted-foreground">Pending requests</span>
              <span className="text-3xl font-bold">{queue?.depth ?? '—'}</span>
            </div>
            <div className="space-y-2">
              {queue ? (
                Object.entries(queue.depth_by_priority).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{key}</span>
                    <span className="font-medium">{value}</span>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground">Collecting metrics...</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* New Transcription form */}
      <Card>
        <CardHeader>
          <CardTitle>New Transcription</CardTitle>
          <CardDescription>Upload a WAV/MP3 file and choose a priority.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="file">Audio file</Label>
              <Input id="file" type="file" accept="audio/*" onChange={onFileChange} />
            </div>
            <div className="space-y-2">
              <Label>Model</Label>
              <Select value={effectiveModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {models?.filter((m) => m.type === 'stt').map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Priority</Label>
              <Select value={priority} onValueChange={setPriority}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PRIORITY_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="language">Language (optional)</Label>
              <Input
                id="language"
                placeholder="en"
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
              />
            </div>
          </div>
          {submitState.error && (
            <p className="text-sm text-destructive">{submitState.error}</p>
          )}
          {submitState.success && (
            <p className="text-sm text-emerald-600">{submitState.success}</p>
          )}
          <Button onClick={onSubmit} disabled={submitState.isSubmitting || !selectedFile}>
            {submitState.isSubmitting ? 'Sending...' : 'Send transcription'}
          </Button>
        </CardContent>
      </Card>

      {/* Jobs table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Jobs</CardTitle>
          <CardDescription>Track status and transcription results.</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Modelo</TableHead>
                <TableHead>Prioridade</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Updated</TableHead>
                <TableHead>Result</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs && jobs.length > 0 ? (
                jobs.map((job) => (
                  <TableRow key={job.request_id}>
                    <TableCell className="font-mono text-xs">{job.request_id.slice(0, 8)}</TableCell>
                    <TableCell>{job.model}</TableCell>
                    <TableCell>{job.priority}</TableCell>
                    <TableCell>
                      <Badge variant={statusVariant(job.status)}>{job.status}</Badge>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(job.updated_at)}
                    </TableCell>
                    <TableCell className="max-w-xs truncate text-sm text-muted-foreground">
                      {job.result?.text ?? job.error ?? 'Waiting...'}
                    </TableCell>
                    <TableCell className="text-right">
                      {job.status === 'queued' && (
                        <Button variant="ghost" size="sm" onClick={() => onCancel(job.request_id)}>
                          Cancel
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={7} className="py-10 text-center">
                    <p className="text-sm font-medium text-muted-foreground">No jobs yet</p>
                    <p className="mt-1 text-xs text-muted-foreground">Upload a file above to get started.</p>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Streaming STT Tab
// ---------------------------------------------------------------------------

interface TranscriptEntry {
  id: number;
  text: string;
  isFinal: boolean;
  timestamp: number;
  isBot?: boolean;
}

function StreamingSTTTab({ models }: { models: DemoModel[] | undefined }) {
  const sttModels = useMemo(() => models?.filter((m) => m.type === 'stt') ?? [], [models]);
  const ttsModels = useMemo(() => models?.filter((m) => m.type === 'tts') ?? [], [models]);
  const [selectedModel, setSelectedModel] = useState('');
  const [language, setLanguage] = useState('en');
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [vadActive, setVadActive] = useState(false);
  const [currentPartial, setCurrentPartial] = useState('');
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [errorRecoverable, setErrorRecoverable] = useState<boolean | undefined>(undefined);

  // TTS state
  const [ttsText, setTtsText] = useState('');
  const [ttsSpeaking, setTtsSpeaking] = useState(false);
  const [sttMuted, setSttMuted] = useState(false);
  const ttsAudioChunksRef = useRef<ArrayBuffer[]>([]);
  const ttsPlaybackCtxRef = useRef<AudioContext | null>(null);
  const lastTtsTextRef = useRef('');

  // Session warnings
  const [sessionHold, setSessionHold] = useState(false);
  const [rateLimitDelay, setRateLimitDelay] = useState<number | null>(null);
  const [framesDropped, setFramesDropped] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const transcriptsEndRef = useRef<HTMLDivElement>(null);

  const effectiveModel = selectedModel || sttModels[0]?.name || '';

  // Auto-scroll transcript
  useEffect(() => {
    transcriptsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcripts, currentPartial]);

  // Play accumulated TTS audio chunks
  const playTtsAudio = useCallback(async () => {
    const chunks = ttsAudioChunksRef.current;
    if (chunks.length === 0) return;

    // Concatenate all chunks into single buffer
    const totalLength = chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }
    ttsAudioChunksRef.current = [];

    // Decode PCM 16-bit 24kHz mono -> float32
    const int16 = new Int16Array(combined.buffer);
    const sampleRate = 24000;
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    try {
      if (!ttsPlaybackCtxRef.current || ttsPlaybackCtxRef.current.state === 'closed') {
        ttsPlaybackCtxRef.current = new AudioContext({ sampleRate });
      }
      const ctx = ttsPlaybackCtxRef.current;
      const audioBuffer = ctx.createBuffer(1, float32.length, sampleRate);
      audioBuffer.copyToChannel(float32, 0);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.start();
    } catch {
      // Playback failed — not critical
    }
  }, []);

  const stopRecording = useCallback(() => {
    processorRef.current?.disconnect();
    processorRef.current = null;

    if (audioContextRef.current?.state !== 'closed') {
      audioContextRef.current?.close();
    }
    audioContextRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'session.close' }));
      wsRef.current.close();
    }
    wsRef.current = null;

    setIsRecording(false);
    setIsConnected(false);
    setVadActive(false);
    setCurrentPartial('');
    setTtsSpeaking(false);
    setSttMuted(false);
    setSessionHold(false);
    setRateLimitDelay(null);
    setFramesDropped(null);
  }, []);

  async function startRecording() {
    if (!effectiveModel) {
      setError('Select an STT model.');
      return;
    }

    setError(null);
    setErrorRecoverable(undefined);
    setTranscripts([]);
    setCurrentPartial('');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const ws = createRealtimeConnection(effectiveModel, language || undefined, {
        onSessionCreated: () => {
          setIsConnected(true);
          setSessionHold(false);
        },
        onVadSpeechStart: () => {
          setVadActive(true);
          setSessionHold(false);
        },
        onVadSpeechEnd: () => {
          setVadActive(false);
        },
        onTranscriptPartial: (text) => {
          setCurrentPartial(text);
        },
        onTranscriptFinal: (text, segmentId) => {
          setCurrentPartial('');
          setTranscripts((prev) => [
            ...prev,
            { id: segmentId, text, isFinal: true, timestamp: Date.now() },
          ]);
        },
        onError: (message, recoverable) => {
          setError(message);
          setErrorRecoverable(recoverable);
        },
        onClose: () => {
          stopRecording();
        },
        // TTS callbacks
        onTtsSpeakingStart: () => {
          setTtsSpeaking(true);
          setSttMuted(true);
          ttsAudioChunksRef.current = [];
        },
        onTtsSpeakingEnd: (_requestId, cancelled) => {
          setTtsSpeaking(false);
          setSttMuted(false);
          if (!cancelled) {
            playTtsAudio();
            setTranscripts((prev) => [
              ...prev,
              {
                id: Date.now(),
                text: lastTtsTextRef.current,
                isFinal: true,
                timestamp: Date.now(),
                isBot: true,
              },
            ]);
          }
          ttsAudioChunksRef.current = [];
        },
        onTtsAudio: (audioData) => {
          ttsAudioChunksRef.current.push(audioData);
        },
        // Session warnings
        onSessionHold: () => {
          setSessionHold(true);
        },
        onRateLimit: (delayMs) => {
          setRateLimitDelay(delayMs);
          setTimeout(() => setRateLimitDelay(null), 5000);
        },
        onFramesDropped: (droppedMs) => {
          setFramesDropped(droppedMs);
          setTimeout(() => setFramesDropped(null), 5000);
        },
      });
      wsRef.current = ws;

      // Wait for WebSocket to open before starting audio
      ws.onopen = () => {
        const audioContext = new AudioContext({ sampleRate: 16000 });
        audioContextRef.current = audioContext;

        const source = audioContext.createMediaStreamSource(stream);
        // Buffer size of 4096 at 16kHz = ~256ms frames
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;

        processor.onaudioprocess = (event) => {
          if (ws.readyState !== WebSocket.OPEN) return;

          const inputData = event.inputBuffer.getChannelData(0);
          // Convert float32 -> int16 PCM
          const pcm16 = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          ws.send(pcm16.buffer);
        };

        source.connect(processor);
        processor.connect(audioContext.destination);
        setIsRecording(true);
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to access microphone';
      setError(message);
      stopRecording();
    }
  }

  function toggleRecording() {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }

  function handleTtsSpeak() {
    if (!wsRef.current || !ttsText.trim()) return;
    lastTtsTextRef.current = ttsText.trim();
    sendTtsSpeak(wsRef.current, ttsText.trim());
    setTtsText('');
  }

  function handleTtsCancel() {
    if (!wsRef.current) return;
    sendTtsCancel(wsRef.current);
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MicIcon className="h-5 w-5" />
            Streaming STT
            {ttsModels.length > 0 && (
              <Badge variant="outline" className="ml-2 text-xs font-normal">Full-Duplex</Badge>
            )}
          </CardTitle>
          <CardDescription>
            Real-time transcription via WebSocket with VAD, partial transcripts{ttsModels.length > 0 ? ' and full-duplex TTS' : ''}.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label>STT model</Label>
              <Select value={effectiveModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {sttModels.map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="stt-lang">Language</Label>
              <Input
                id="stt-lang"
                placeholder="en"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              />
            </div>
            <div className="flex items-end">
              <Button
                onClick={toggleRecording}
                variant={isRecording ? 'destructive' : 'default'}
                className="w-full gap-2"
              >
                {isRecording ? (
                  <>
                    <StopIcon className="h-4 w-4" />
                    Stop recording
                  </>
                ) : (
                  <>
                    <MicIcon className="h-4 w-4" />
                    Start recording
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Status indicators */}
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`h-2.5 w-2.5 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-muted-foreground/30'}`} />
              <span className="text-xs text-muted-foreground">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="relative">
                <div className={`h-2.5 w-2.5 rounded-full ${vadActive ? 'bg-red-500' : 'bg-muted-foreground/30'}`} />
                {vadActive && (
                  <div className="animate-pulse-ring absolute inset-0 h-2.5 w-2.5 rounded-full bg-red-500" />
                )}
              </div>
              <span className="text-xs text-muted-foreground">
                {vadActive ? 'Speech detected' : 'Silence'}
              </span>
            </div>
            {sttMuted && (
              <Badge variant="warning" className="gap-1 text-xs">
                <VolumeOffIcon className="h-3 w-3" />
                STT muted — TTS active
              </Badge>
            )}
            {sessionHold && (
              <Badge variant="warning" className="text-xs">
                Session on hold
              </Badge>
            )}
            {rateLimitDelay !== null && (
              <Badge variant="warning" className="text-xs">
                Rate limit: {rateLimitDelay}ms
              </Badge>
            )}
            {framesDropped !== null && (
              <Badge variant="destructive" className="text-xs">
                Frames dropped: {framesDropped}ms
              </Badge>
            )}
            {isRecording && !sttMuted && (
              <div className="flex items-center gap-1">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="animate-waveform w-1 rounded-full bg-primary"
                    style={{ animationDelay: `${i * 0.15}s`, height: '4px' }}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Error banner */}
          {error && (
            <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-3">
              <span className="mt-0.5 text-destructive">!</span>
              <div className="flex-1">
                <p className="text-sm text-destructive">{error}</p>
                {errorRecoverable !== undefined && (
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {errorRecoverable ? 'Recoverable error — the session will continue.' : 'Unrecoverable error.'}
                  </p>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Transcript output */}
      <Card>
        <CardHeader>
          <CardTitle>Transcription</CardTitle>
          <CardDescription>
            {transcripts.filter((t) => !t.isBot).length} confirmed segment{transcripts.filter((t) => !t.isBot).length !== 1 ? 's' : ''}
            {transcripts.some((t) => t.isBot) && ` · ${transcripts.filter((t) => t.isBot).length} TTS response${transcripts.filter((t) => t.isBot).length !== 1 ? 's' : ''}`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-h-96 min-h-[120px] space-y-2 overflow-y-auto rounded-lg bg-muted/30 p-4">
            {transcripts.length === 0 && !currentPartial && (
              <p className="text-center text-sm text-muted-foreground">
                {isRecording
                  ? 'Waiting for speech...'
                  : 'Click "Start recording" to begin.'}
              </p>
            )}

            {transcripts.map((entry) => (
              <div key={`${entry.id}-${entry.timestamp}`} className="fade-slide-in flex items-start gap-2">
                {entry.isBot ? (
                  <Badge variant="default" className="mt-0.5 shrink-0 text-xs">
                    Bot
                  </Badge>
                ) : (
                  <Badge variant="success" className="mt-0.5 shrink-0 text-xs">
                    Final
                  </Badge>
                )}
                <p className={`text-sm ${entry.isBot ? 'text-muted-foreground' : ''}`}>
                  {entry.isBot ? `"${entry.text}"` : entry.text}
                </p>
              </div>
            ))}

            {currentPartial && (
              <div className="fade-slide-in flex items-start gap-2">
                <Badge variant="outline" className="mt-0.5 shrink-0 text-xs">
                  Partial
                </Badge>
                <p className="text-sm italic text-muted-foreground">{currentPartial}</p>
              </div>
            )}

            <div ref={transcriptsEndRef} />
          </div>
        </CardContent>
      </Card>

      {/* TTS Full-Duplex input */}
      {ttsModels.length > 0 && (
        <Card className={sttMuted ? 'mute-indicator' : ''}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <VolumeIcon className="h-5 w-5" />
              TTS Full-Duplex
              {ttsSpeaking && (
                <Badge variant="default" className="ml-2 gap-1 text-xs">
                  <div className="flex items-center gap-0.5">
                    {[0, 1, 2].map((i) => (
                      <div
                        key={i}
                        className="animate-waveform w-0.5 rounded-full bg-primary-foreground"
                        style={{ animationDelay: `${i * 0.15}s`, height: '3px' }}
                      />
                    ))}
                  </div>
                  Speaking...
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Send text for the bot to speak via WebSocket (tts.speak). STT is automatically muted during synthesis.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Input
                placeholder={isConnected ? 'Type the text for the bot to speak...' : 'Connect streaming first'}
                value={ttsText}
                onChange={(e) => setTtsText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleTtsSpeak();
                  }
                }}
                disabled={!isConnected}
              />
              <Button
                onClick={handleTtsSpeak}
                disabled={!isConnected || !ttsText.trim() || ttsSpeaking}
                className="gap-2 shrink-0"
              >
                <SendIcon className="h-4 w-4" />
                Speak
              </Button>
              {ttsSpeaking && (
                <Button
                  onClick={handleTtsCancel}
                  variant="destructive"
                  size="sm"
                  className="shrink-0"
                >
                  Cancel
                </Button>
              )}
            </div>
            {!isConnected && (
              <p className="text-xs text-muted-foreground">
                Start recording above to enable full-duplex TTS on the same WebSocket connection.
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TTS Playback Tab
// ---------------------------------------------------------------------------

function TTSPlaybackTab({ models }: { models: DemoModel[] | undefined }) {
  const ttsModels = useMemo(() => models?.filter((m) => m.type === 'tts') ?? [], [models]);
  const [selectedModel, setSelectedModel] = useState('');
  const [text, setText] = useState('');
  const [voice, setVoice] = useState('default');
  const [speed, setSpeed] = useState(1.0);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ttfbMs, setTtfbMs] = useState<number | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);

  const effectiveModel = selectedModel || ttsModels[0]?.name || '';

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  async function onSynthesize() {
    if (!text.trim()) {
      setError('Enter text to synthesize.');
      return;
    }
    if (!effectiveModel) {
      setError('No TTS model available.');
      return;
    }

    setError(null);
    setIsSynthesizing(true);
    setTtfbMs(null);

    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }

    const startTime = performance.now();

    try {
      const blob = await synthesizeSpeech({
        model: effectiveModel,
        input: text.trim(),
        voice,
        speed,
      });

      const elapsed = Math.round(performance.now() - startTime);
      setTtfbMs(elapsed);

      const url = URL.createObjectURL(blob);
      setAudioUrl(url);

      // Auto-play
      setTimeout(() => {
        audioRef.current?.play();
      }, 50);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Synthesis error';
      setError(message);
    } finally {
      setIsSynthesizing(false);
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <VolumeIcon className="h-5 w-5" />
            Text-to-Speech
          </CardTitle>
          <CardDescription>
            Synthesize speech from text via POST /v1/audio/speech (OpenAI-compatible).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="tts-text">Text</Label>
            <textarea
              id="tts-text"
              className="flex min-h-[100px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              placeholder="Type the text you want to synthesize..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label>TTS model</Label>
              <Select value={effectiveModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {ttsModels.map((model) => (
                    <SelectItem key={model.name} value={model.name}>
                      {model.name}
                    </SelectItem>
                  ))}
                  {ttsModels.length === 0 && (
                    <SelectItem value="_none" disabled>
                      No TTS model installed
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="tts-voice">Voice</Label>
              <Input
                id="tts-voice"
                placeholder="default"
                value={voice}
                onChange={(e) => setVoice(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Speed: {speed.toFixed(1)}x</Label>
              <Slider
                min={0.5}
                max={2.0}
                step={0.1}
                value={[speed]}
                onValueChange={([v]) => setSpeed(v)}
              />
            </div>
          </div>

          {ttsModels.length === 0 && (
            <div className="rounded-lg border border-dashed p-4 text-center">
              <p className="text-sm font-medium">No TTS model installed</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Run <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">macaw pull kokoro-v1</code> to enable TTS.
              </p>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-3">
              <span className="mt-0.5 text-destructive">!</span>
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}

          <div className="flex items-center gap-4">
            <Button
              onClick={onSynthesize}
              disabled={isSynthesizing || !text.trim() || ttsModels.length === 0}
              className="gap-2"
            >
              <VolumeIcon className="h-4 w-4" />
              {isSynthesizing ? 'Synthesizing...' : 'Synthesize speech'}
            </Button>

            {ttfbMs !== null && (
              <Badge variant="outline" className="text-xs">
                TTFB: {ttfbMs}ms
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Audio player */}
      {audioUrl && (
        <Card>
          <CardHeader>
            <CardTitle>Result</CardTitle>
            <CardDescription>Synthesized audio — click play to listen.</CardDescription>
          </CardHeader>
          <CardContent>
            <audio
              ref={audioRef}
              controls
              src={audioUrl}
              className="w-full"
            />
          </CardContent>
        </Card>
      )}

      {/* Architecture info */}
      <Card className="border-dashed">
        <CardContent className="py-6">
          <div className="grid gap-4 text-sm md:grid-cols-3">
            <div>
              <p className="font-semibold">Endpoint</p>
              <p className="text-muted-foreground">POST /v1/audio/speech</p>
            </div>
            <div>
              <p className="font-semibold">Format</p>
              <p className="text-muted-foreground">WAV (PCM 16-bit) or raw PCM</p>
            </div>
            <div>
              <p className="font-semibold">Streaming (WebSocket)</p>
              <p className="text-muted-foreground">tts.speak / tts.cancel with mute-on-speak</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function Page() {
  const { data: models } = useSWR<DemoModel[]>('models', fetchModels);
  const { data: queue } = useSWR<QueueMetrics>('queue', fetchQueue, {
    refreshInterval: 3000,
  });
  const { data: jobs } = useSWR<DemoJob[]>('jobs', fetchJobs, {
    refreshInterval: 4000,
  });

  const defaultModel = useMemo(() => {
    const sttModel = models?.find((m) => m.type === 'stt');
    return sttModel?.name ?? models?.[0]?.name ?? '';
  }, [models]);

  return (
    <main className="min-h-screen bg-background">
      <section className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-6 py-10">
        <HeroSection modelCount={models?.length ?? 0} />

        <Tabs defaultValue="dashboard" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="dashboard" className="gap-2">
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="streaming" className="gap-2">
              <MicIcon className="h-4 w-4" />
              Streaming
            </TabsTrigger>
            <TabsTrigger value="tts" className="gap-2">
              <VolumeIcon className="h-4 w-4" />
              Text-to-Speech
            </TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard">
            <DashboardTab
              models={models}
              queue={queue}
              jobs={jobs}
              defaultModel={defaultModel}
            />
          </TabsContent>

          <TabsContent value="streaming">
            <StreamingSTTTab models={models} />
          </TabsContent>

          <TabsContent value="tts">
            <TTSPlaybackTab models={models} />
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <footer className="border-t pt-6 text-center text-xs text-muted-foreground">
          <p>
            Macaw OpenVoice — Voice runtime (STT + TTS).
            All 3 PRD phases delivered (M1-M9). 1600+ tests.
          </p>
          <p className="mt-1">
            OpenAI-compatible API · gRPC worker isolation · Full-duplex WebSocket · Mute-on-speak
          </p>
        </footer>
      </section>
    </main>
  );
}
