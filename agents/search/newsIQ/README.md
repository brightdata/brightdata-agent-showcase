# NewsIQ - AI-Powered News Research Assistant

An intelligent news research assistant that searches current news, scrapes articles, and provides unbiased analysis using AI and real-time web data.

## Features

- **Real-time News Search**: Search Google News for current events
- **Article Scraping**: Extract full article content bypassing paywalls
- **AI Analysis**: GPT-4o-mini analyzes news with citations and insights
- **Multi-Source Research**: Compare coverage across different sources
- **Streaming Responses**: Real-time AI responses with tool execution

## Prerequisites

- Node.js 18+
- [OpenAI API Key](https://platform.openai.com/api-keys)
- [Bright Data API Key](https://brightdata.com/)

## Setup

1. Clone and install dependencies:
```bash
npm install
```

2. Create `.env.local` with your API keys:
```env
OPENAI_API_KEY=your_openai_key
BRIGHTDATA_API_KEY=your_brightdata_key
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000)

## Tech Stack

- Next.js 16 + React 19
- AI SDK v5 (Vercel)
- Bright Data SDK
- Tailwind CSS 4
