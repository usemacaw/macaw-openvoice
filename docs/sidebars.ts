import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/configuration',
      ],
    },
    {
      type: 'category',
      label: 'Supported Models',
      items: [
        'models/index',
        'models/faster-whisper',
        'models/wenet',
        'models/kokoro',
        'models/silero-vad',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/batch-transcription',
        'guides/streaming-stt',
        'guides/full-duplex',
        'guides/adding-engine',
        'guides/cli',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api-reference/rest-api',
        'api-reference/websocket-protocol',
        'api-reference/grpc-internal',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/overview',
        'architecture/session-manager',
        'architecture/vad-pipeline',
        'architecture/scheduling',
      ],
    },
    {
      type: 'category',
      label: 'Community',
      items: [
        'community/contributing',
        'community/changelog',
        'community/roadmap',
      ],
    },
  ],
};

export default sidebars;
