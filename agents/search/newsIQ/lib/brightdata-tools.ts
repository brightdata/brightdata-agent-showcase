import { tool, type Tool } from "ai";
import { z } from "zod";
import { bdclient } from "@brightdata/sdk";

type NewsTools = "searchNews" | "scrapeArticle" | "searchWeb";

interface NewsToolsConfig {
  apiKey: string;
  excludeTools?: NewsTools[];
}

export const newsTools = (
  config: NewsToolsConfig
): Partial<Record<NewsTools, Tool>> => {
  const client = new bdclient({
    apiKey: config.apiKey,
    autoCreateZones: true,
  });

  const tools: Partial<Record<NewsTools, Tool>> = {
    searchNews: tool({
      description:
        "Search for news articles on any topic using Google News. Returns recent news articles with titles, snippets, sources, and publication dates. Use this for finding current news coverage on specific topics.",
      inputSchema: z.object({
        query: z
          .string()
          .describe(
            'The news search query (e.g., "artificial intelligence", "climate change policy", "tech earnings")'
          ),
        country: z
          .string()
          .length(2)
          .optional()
          .describe(
            'Two-letter country code for localized news (e.g., "us", "gb", "de", "fr", "jp")'
          ),
      }),
      execute: async ({
        query,
        country,
      }: {
        query: string;
        country?: string;
      }) => {
        try {
          const newsQuery = `${query} news`;
          const result = await client.search(newsQuery, {
            searchEngine: "google",
            dataFormat: "markdown",
            format: "raw",
            country: country?.toLowerCase() || "us",
          });
          return result;
        } catch (error) {
          return `Error searching for news on "${query}": ${String(error)}`;
        }
      },
    }),

    scrapeArticle: tool({
      description:
        "Scrape the full content of a news article from any URL. Returns the complete article text in clean markdown format, bypassing paywalls and anti-bot protection. Use this to read full articles after finding them with searchNews.",
      inputSchema: z.object({
        url: z.string().url().describe("The URL of the news article to scrape"),
        country: z
          .string()
          .length(2)
          .optional()
          .describe("Two-letter country code for proxy location"),
      }),
      execute: async ({ url, country }: { url: string; country?: string }) => {
        try {
          const result = await client.scrape(url, {
            dataFormat: "markdown",
            format: "raw",
            country: country?.toLowerCase(),
          });
          return result;
        } catch (error) {
          return `Error scraping article at ${url}: ${String(error)}`;
        }
      },
    }),

    searchWeb: tool({
      description:
        "General web search using Google, Bing, or Yandex. Use this for background research, fact-checking, or finding additional context beyond news articles.",
      inputSchema: z.object({
        query: z
          .string()
          .describe(
            "The search query for background information or fact-checking"
          ),
        searchEngine: z
          .enum(["google", "bing", "yandex"])
          .optional()
          .default("google")
          .describe("Search engine to use"),
        country: z
          .string()
          .length(2)
          .optional()
          .describe("Two-letter country code for localized results"),
      }),
      execute: async ({
        query,
        searchEngine,
        country,
      }: {
        query: string;
        searchEngine: "google" | "bing" | "yandex";
        country?: string;
      }) => {
        try {
          const result = await client.search(query, {
            searchEngine,
            dataFormat: "markdown",
            format: "raw",
            country: country?.toLowerCase(),
          });
          return result;
        } catch (error) {
          return `Error searching for "${query}": ${String(error)}`;
        }
      },
    }),
  };

  // Remove excluded tools if specified
  for (const toolName in tools) {
    if (config.excludeTools?.includes(toolName as NewsTools)) {
      delete tools[toolName as NewsTools];
    }
  }

  return tools;
};
