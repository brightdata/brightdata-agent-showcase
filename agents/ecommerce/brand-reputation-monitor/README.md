# Brand Reputation Monitoring Workflow

AI-powered brand reputation monitoring that searches Google News, analyzes sentiment, and delivers insights via email.

<div align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue"/>
  <img src="https://img.shields.io/badge/License-MIT-blue"/>
</div>

---

## Tech stack

- **Bright Data SDK**
- **OpenAI GPT**
- **SendGrid**
- **Python**
- **Pydantic**

---

## Features

- **Automated News Discovery**: Uses [Bright Data's SERP API](https://brightdata.com/products/serp-api) to find Google News pages for your brand queries.
- **Intelligent Content Extraction**: Scrapes news articles at scale using [Bright Data's Web Unlocker API](https://brightdata.com/products/web-unlocker) with parallel processing.
- **AI-Powered Analysis**: Leverages OpenAI GPT-5-mini to analyze sentiment, extract insights, and identify the most relevant news for brand monitoring.
- **Smart Content Filtering**: Automatically selects the top news articles most relevant to your brand reputation.
- **Professional Email Reports**: Generates beautifully formatted HTML email reports with sentiment analysis and actionable insights.
- **Automated Delivery**: Sends monitoring reports directly to stakeholders via SendGrid.
- **Configurable Monitoring**: Easy-to-customize search queries and recipient lists through JSON configuration.

---

## How It Works 🔄

1. **Configuration Loading**: Reads your monitoring setup from `config.json`
2. **News Discovery**: Searches Google SERPs using your brand queries to find Google News page URLs
3. **Content Extraction**: Scrapes all relevant news pages in parallel and gets Markdown content
4. **Content Selection**: AI identifies the most important news articles from the scraped pages
5. **Individual Article Scraping**: Scrapes each selected news article for detailed content
6. **Detailed Analysis**: Each article gets sentiment analysis and brand insights
7. **Report Generation**: Creates a professional HTML email report
8. **Automated Delivery**: Sends the report to your team via email

---

## Prerequisites 🛠️

- Python 3.8+ 🐍
- [Bright Data API token](https://docs.brightdata.com/api-reference/authentication) 🔑
- OpenAI API key 🔑
- SendGrid API key 🔑

---

## Installation ⚙️

1. Clone this repository:
   ```bash
   git clone https://github.com/brightdata/brand-reputation-monitoring-workflow
   cd brand-reputation-monitoring-workflow
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   ```
   On Linux/macOS, activate it with:
   ```bash
   source .venv/bin/activate
   ```
   On Windows, run:
   ```bash
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install python-dotenv brightdata-sdk openai sendgrid pydantic
   ```
4. Create a `.env` file in the project root with your API keys:
   ```
   BRIGHT_DATA_API_TOKEN=your_bright_data_api_token
   OPENAI_API_KEY=your_openai_api_key
   SENDGRID_API_KEY=your_sendgrid_api_key
   ```

---

## Configuration 📝

Create a `config.json` in the root directory to customize the brand monitoring workflow:

```json
{
  "search_queries": [
    "your_brand news",
    "your_brand reviews",
    "your_brand controversy",
    "your_company announcement"
  ],
  "num_news": 5,
  "sender": "monitoring@yourcompany.com",
  "recipients": [
    "pr@yourcompany.com",
    "marketing@yourcompany.com",
    "ceo@yourcompany.com"
  ]
}
```

**Configuration Fields**:

- `search_queries`: List of search terms to monitor your brand (supports multiple queries)
- `num_news`: Number of top articles to analyze in detail (default: `5`)
- `sender`: Email address to send reports from (must be verified in SendGrid)
- `recipients`: List of email addresses to receive monitoring reports

---

## Project Structure 📁

```
brand-reputation-monitoring-workflow/
├── .venv/
├── .env
├── config.json
└── workflow.py
```

---

## Usage ▶️

Run the brand monitoring workflow with:

```
python workflow.py
```

The workflow will:

1. 🔍 Search Google SERPs for your configured queries and extract Google News URLs
2. 📰 Scrape Google News pages to get all available article URLs
3. 🤖 Use AI to select the most relevant articles for brand monitoring
4. 📄 Scrape individual news articles for detailed content
5. 🧠 Generate AI-powered insights and sentiment analysis for each article
6. 📧 Send a professional HTML report to your team

**Sample Output:**

```
Retrieving Google News page URLs for the following search queries: nike, nike shoes
2 Google News page URL(s) retrieved!

Scraping content from each Google News page...
Google News pages scraped!

Extracting the most relevant news URLs...
5 news articles found:
- https://www.espn.com/wnba/story/_/id/46075454/caitlin-clark-becomes-nike-newest-signature-athlete
- https://wwd.com/footwear-news/sneaker-news/nike-acg-radical-airflow-ultrafly-release-dates-1238068936/
- https://www.runnersworld.com/news/a65881486/cooper-lutkenhaus-professional-contract-nike/
- https://hypebeast.com/2025/8/nike-kobe-3-protro-low-reveal-info
- https://wwd.com/footwear-news/sneaker-news/nike-air-diamond-turf-must-be-the-money-release-date-1238075256/

Scraping the selected news articles...
5 news articles scraped!

Analyzing each news for brand reputation monitoring...
News analysis complete!

Generating HTML email body...
HTML email body generated!

Sending the email with the brand reputation monitoring HTML report...
Email sent!
```

---

## Email Report Features 📊

Each automated report includes:

- **Article Summaries**: Concise 30-word summaries of each news piece
- **Sentiment Analysis**: Positive, negative, or neutral sentiment classification with color-coded labels
- **Actionable Insights**: 3-5 key takeaways for brand reputation management (10-12 words each)
- **Direct Links**: Easy access to original articles for deeper review
- **Professional Formatting**: Clean, responsive HTML design ready for stakeholders

---

## Advanced Configuration 🧑‍💻

### Custom Search Parameters

Modify search queries for different monitoring scenarios, as well as configure senders, recipients, and the number of news items in `config.json`.

### Scheduling Automation

Set up automated monitoring with cron jobs:

```bash
# Run every Monday at 9 AM
0 9 * * 1 /usr/bin/python3 /path/to/your/project/workflow.py
```

### Custom Analysis Prompts

Fine-tune the AI analysis by modifying the system prompts in the `process_news_list()` function for industry-specific insights.

### Next Steps & Enhancements

- **Add memory layer**: Avoid analyzing the same articles multiple times
- **SendGrid templating**: Use consistent email templates for standardized reports
- **Cloud storage**: Archive reports in S3 for historical analysis

---

## API Integration Details 🔧

This workflow leverages powerful APIs through the Bright Data SDK:

- **[Bright Data SERP API](https://brightdata.com/products/serp-api)**: For discovering Google News URLs from search results
- **[Bright Data Web Unlocker API](https://brightdata.com/products/web-unlocker)**: For parallel content extraction in LLM-optimized Markdown format
- **OpenAI GPT-5-mini**: For content analysis and HTML report generation
- **SendGrid Email API**: For professional report delivery

---

## Troubleshooting & Tips 💡

- **API Keys**: Ensure all API keys are correctly set in your `.env` file
- **Email Verification**: Sender email must be verified in SendGrid dashboard
- **Search Queries**: Use specific brand terms and variations for comprehensive monitoring
- **Content Quality**: More specific queries yield better analysis results

### Common Issues:

- **No news found**: Try broader search queries or check if your brand has recent coverage
- **Email not delivered**: Verify sender email in SendGrid and check recipient spam folders
- **Analysis quality**: Refine search queries to get more relevant articles
- **403 Forbidden error**: Ensure sender email is verified in SendGrid account

---

## Use Cases 🎯

- **Crisis Management**: Early detection of negative brand mentions
- **Competitive Intelligence**: Monitor competitor news and market positioning
- **PR Campaign Tracking**: Measure coverage and sentiment of marketing initiatives
- **Product Launch Monitoring**: Track reception and feedback on new releases
- **Executive Briefings**: Regular brand health reports for leadership teams

---

**Stay ahead of your brand reputation! 🚀**

Built with ❤️ using [Bright Data's AI infrastructure](https://brightdata.com/ai) for live web data solutions.
