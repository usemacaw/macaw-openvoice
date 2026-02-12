import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Macaw OpenVoice',
  tagline: 'Real-time Speech-to-Text and Text-to-Speech with OpenAI-compatible API, streaming session control, and extensible execution architecture',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://macawvoice.github.io',
  baseUrl: '/macaw-openvoice/',

  organizationName: 'usemacaw',
  projectName: 'macaw-openvoice',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/usemacaw/macaw-openvoice/edit/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      require.resolve('@cmfcmf/docusaurus-search-local'),
      {
        indexDocs: true,
        indexBlog: false,
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: false,
    },
    announcementBar: {
      id: 'star_us',
      content: 'If you like Macaw OpenVoice, give us a <a target="_blank" rel="noopener noreferrer" href="https://github.com/usemacaw/macaw-openvoice">star on GitHub</a>!',
      backgroundColor: '#FDD614',
      textColor: '#0e1017',
      isCloseable: true,
    },
    navbar: {
      title: 'Macaw OpenVoice',
      logo: {
        alt: 'Macaw OpenVoice',
        src: 'img/logo-64.png',
      },
      items: [
        {
          type: 'doc',
          docId: 'intro',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'doc',
          docId: 'getting-started/quickstart',
          position: 'left',
          label: 'Quickstart',
        },
        {
          type: 'doc',
          docId: 'api-reference/rest-api',
          position: 'left',
          label: 'API',
        },
        {
          type: 'doc',
          docId: 'architecture/overview',
          position: 'left',
          label: 'Architecture',
        },
        {
          href: 'https://github.com/usemacaw/macaw-openvoice',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {
              label: 'Welcome',
              to: '/docs/intro',
            },
            {
              label: 'Quickstart',
              to: '/docs/getting-started/quickstart',
            },
            {
              label: 'Installation',
              to: '/docs/getting-started/installation',
            },
          ],
        },
        {
          title: 'Guides',
          items: [
            {
              label: 'Streaming STT',
              to: '/docs/guides/streaming-stt',
            },
            {
              label: 'Full-Duplex',
              to: '/docs/guides/full-duplex',
            },
            {
              label: 'Adding an Engine',
              to: '/docs/guides/adding-engine',
            },
          ],
        },
        {
          title: 'Reference',
          items: [
            {
              label: 'REST API',
              to: '/docs/api-reference/rest-api',
            },
            {
              label: 'WebSocket Protocol',
              to: '/docs/api-reference/websocket-protocol',
            },
            {
              label: 'Architecture',
              to: '/docs/architecture/overview',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/usemacaw/macaw-openvoice',
            },
            {
              label: 'Issues',
              href: 'https://github.com/usemacaw/macaw-openvoice/issues',
            },
            {
              label: 'Contributing',
              to: '/docs/community/contributing',
            },
          ],
        },
      ],
      copyright: `Copyright \u00A9 ${new Date().getFullYear()} Macaw Team. Apache 2.0 License.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'json', 'yaml', 'toml', 'protobuf'],
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
