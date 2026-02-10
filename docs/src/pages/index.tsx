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
      <div className={styles.heroInner}>
        <div className={styles.heroCopy}>
          <p className={styles.heroEyebrow}>Macaw OpenVoice</p>
          <Heading as="h1" className={styles.heroTitle}>
            Runtime for streaming speech.
          </Heading>
          <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
          <div className={styles.heroActions}>
            <Link className="button button--primary button--lg" to="/docs/getting-started/quickstart">
              Get Started
            </Link>
            <Link className="button button--outline button--lg" to="https://github.com/useMacaw/Macaw-openvoice">
              View on GitHub
            </Link>
          </div>
        </div>
        <div className={styles.heroCard}>
          <p className={styles.heroCardTitle}>Quick start</p>
          <pre className={styles.heroCode}>
            <code>{`pip install Macaw-openvoice[server,grpc,faster-whisper]

Macaw serve

curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=faster-whisper-large-v3`}</code>
          </pre>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Voice runtime (STT + TTS) with OpenAI-compatible API">
      <HomepageHeader />
      <main>
        <section className={styles.section}>
          <div className="container">
            <div className={styles.sectionIntro}>
              <Heading as="h2" className={styles.sectionTitle}>
                Built for real-time voice pipelines
              </Heading>
              <p className={styles.sectionSubtitle}>
                A single runtime that handles streaming STT, TTS, VAD, preprocessing, and scheduling.
              </p>
            </div>
          </div>
        </section>
        <HomepageFeatures />
        <section className={styles.sectionAlt}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              Architecture snapshot
            </Heading>
            <p className={styles.sectionSubtitle}>
              STT and TTS share a single server with isolated gRPC workers per engine.
            </p>
            <pre className={styles.archDiagram}>
              <code>{`Clients -> API Server -> Scheduler -> gRPC Workers
                         |              |
                         |              +-> TTS (Kokoro)
                         +-> STT (Faster-Whisper, WeNet)`}</code>
            </pre>
          </div>
        </section>
      </main>
    </Layout>
  );
}
