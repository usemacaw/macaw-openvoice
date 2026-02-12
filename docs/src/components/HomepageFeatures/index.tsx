import type {ReactNode} from 'react';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  icon: string;
  description: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Streaming STT',
    icon: '\uD83C\uDF99\uFE0F',
    description: 'Real-time partial and final transcripts via WebSocket with sub-300ms TTFB and backpressure control.',
  },
  {
    title: 'Text-to-Speech',
    icon: '\uD83D\uDD0A',
    description: 'OpenAI-compatible speech endpoint with streaming PCM or WAV output and low time-to-first-byte.',
  },
  {
    title: 'Full-Duplex',
    icon: '\u21C4',
    description: 'Simultaneous STT and TTS on one WebSocket connection with automatic mute-on-speak safety.',
  },
  {
    title: 'Session Manager',
    icon: '\uD83D\uDEE1\uFE0F',
    description: '6-state machine with ring buffer, WAL-based crash recovery, and zero segment duplication.',
  },
  {
    title: 'Multi-Engine',
    icon: '\u2699\uFE0F',
    description: 'Faster-Whisper, WeNet, and Kokoro through a single interface. Add new engines in ~500 lines.',
  },
  {
    title: 'Voice Pipeline',
    icon: '\uD83D\uDD17',
    description: 'Preprocessing, Silero VAD, ITN post-processing, and Prometheus metrics — all built in.',
  },
];

function Feature({title, icon, description}: FeatureItem) {
  return (
    <div className={styles.featureCard}>
      <div className={styles.featureIcon}>{icon}</div>
      <Heading as="h3" className={styles.featureTitle}>
        {title}
      </Heading>
      <p className={styles.featureDescription}>{description}</p>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featuresHeader}>
          <Heading as="h2" className={styles.featuresTitle}>
            Everything you need for voice
          </Heading>
          <p className={styles.featuresSubtitle}>
            A single runtime that handles the entire voice pipeline — from raw audio to structured text and back.
          </p>
        </div>
        <div className={styles.featureGrid}>
          {FeatureList.map((feature) => (
            <Feature key={feature.title} {...feature} />
          ))}
        </div>
      </div>
    </section>
  );
}
