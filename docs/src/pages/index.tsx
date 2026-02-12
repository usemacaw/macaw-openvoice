import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <header className={styles.heroBanner}>
      <div className={styles.heroGlow} />
      <div className={styles.heroInner}>
        <div className={styles.heroCopy}>
          <p className={styles.heroEyebrow}>Open-Source Voice Runtime</p>
          <Heading as="h1" className={styles.heroTitle}>
            Build voice apps{' '}
            <span className={styles.heroHighlight}>in minutes,</span>{' '}
            not months
          </Heading>
          <p className={styles.heroSubtitle}>
            Macaw OpenVoice is a production-ready runtime for real-time speech-to-text and
            text-to-speech. Drop-in OpenAI API compatibility, streaming WebSocket support,
            and multi-engine architecture — all in a single Python process.
          </p>
          <div className={styles.heroActions}>
            <Link className="button button--primary button--lg" to="/docs/getting-started/quickstart">
              Get Started
            </Link>
            <Link className="button button--outline button--lg" to="/docs/intro">
              Read the Docs
            </Link>
          </div>
          <div className={styles.heroMeta}>
            <span className={styles.heroBadge}>Python 3.11+</span>
            <span className={styles.heroBadge}>Apache 2.0</span>
            <span className={styles.heroBadge}>1600+ tests</span>
          </div>
        </div>
        <div className={styles.heroCard}>
          <div className={styles.heroCardHeader}>
            <span className={styles.heroCardDot} />
            <span className={styles.heroCardDot} />
            <span className={styles.heroCardDot} />
            <span className={styles.heroCardLabel}>terminal</span>
          </div>
          <pre className={styles.heroCode}>
            <code>{`$ pip install macaw-openvoice[server,grpc,faster-whisper]

$ macaw serve
  ╔═══════════════════════════════════════╗
  ║       Macaw OpenVoice v1.0.0         ║
  ╚═══════════════════════════════════════╝
  INFO  Found 2 model(s)
  INFO  STT worker ready   port=50051
  INFO  TTS worker ready   port=50052
  INFO  Uvicorn running on http://127.0.0.1:8000

$ curl -X POST localhost:8000/v1/audio/transcriptions \\
    -F file=@audio.wav -F model=faster-whisper-tiny

{"text": "Hello, how can I help you today?"}`}</code>
          </pre>
        </div>
      </div>
    </header>
  );
}

function CompatibilitySection() {
  return (
    <section className={styles.compatSection}>
      <div className="container">
        <div className={styles.compatGrid}>
          <div className={styles.compatCopy}>
            <Heading as="h2" className={styles.sectionTitle}>
              OpenAI SDK compatible
            </Heading>
            <p className={styles.sectionSubtitle}>
              Existing OpenAI client libraries work out of the box.
              Just point <code>base_url</code> to your Macaw server.
            </p>
          </div>
          <div className={styles.compatCode}>
            <pre className={styles.heroCode}>
              <code>{`from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

result = client.audio.transcriptions.create(
    model="faster-whisper-tiny",
    file=open("audio.wav", "rb"),
)
print(result.text)`}</code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}

function ArchitectureSection() {
  return (
    <section className={styles.archSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Architecture at a glance
        </Heading>
        <p className={styles.sectionSubtitle}>
          A single runtime orchestrates isolated gRPC workers per engine.
          Workers crash independently — the runtime recovers automatically.
        </p>
        <pre className={styles.archDiagram}>
          <code>{`              Clients (REST / WebSocket / CLI)
                          │
              ┌───────────┴───────────┐
              │     API Server        │
              │  (FastAPI + Uvicorn)  │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │      Scheduler        │
              │  Priority · Batching  │
              │  Cancellation · TTFB  │
              └─────┬─────────┬───────┘
                    │         │
           ┌────────┴──┐  ┌───┴────────┐
           │ STT Worker │  │ TTS Worker │
           │  (gRPC)    │  │  (gRPC)    │
           ├────────────┤  ├────────────┤
           │ Faster-    │  │ Kokoro     │
           │ Whisper    │  └────────────┘
           │ WeNet      │
           └────────────┘`}</code>
        </pre>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title="Build voice apps in minutes"
      description="Real-time Speech-to-Text and Text-to-Speech with OpenAI-compatible API, streaming session control, and extensible execution architecture.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <CompatibilitySection />
        <ArchitectureSection />
      </main>
    </Layout>
  );
}
