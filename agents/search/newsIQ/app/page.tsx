"use client";

import { useChat } from "@ai-sdk/react";
import { useState } from "react";

export default function NewsResearchAssistant() {
  const { messages, sendMessage, status } = useChat();
  const [input, setInput] = useState("");

  // Debug: Log messages to console
  console.log("Messages:", messages);
  console.log("Status:", status);
  if (messages.length > 0) {
    console.log("First message parts:", messages[0].parts);
    if (messages.length > 1) {
      console.log("Second message parts:", messages[1].parts);
    }
  }

  const [exampleQueries] = useState([
    "🌍 What are the latest developments in climate change policy?",
    "💻 Search for news about artificial intelligence regulation",
    "📊 How are different sources covering the economy?",
    "⚡ What are the trending tech stories this week?",
    "🔍 Fact-check: Did [specific claim] really happen?",
  ]);

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-md border-b border-gray-200">
        <div className="max-w-5xl mx-auto px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-600 to-indigo-600 w-12 h-12 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-2xl">📰</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">NewsIQ</h1>
              <p className="text-sm text-gray-600">
                AI-Powered News Research & Analysis
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Chat Area */}
      <div className="flex-1 overflow-hidden max-w-5xl w-full mx-auto px-6 py-6">
        <div className="h-full flex flex-col bg-white rounded-2xl shadow-xl border border-gray-200">
          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center px-4">
                {/* Welcome Screen */}
                <div className="bg-gradient-to-br from-blue-500 to-indigo-600 w-20 h-20 rounded-2xl flex items-center justify-center mb-6 shadow-lg">
                  <span className="text-4xl">📰</span>
                </div>
                <h2 className="text-3xl font-bold text-gray-900 mb-3">
                  Welcome to NewsIQ
                </h2>
                <p className="text-gray-600 mb-8 max-w-2xl text-lg">
                  Your AI-powered research assistant for news analysis,
                  fact-checking, and staying informed. I can search across news
                  sources, analyze bias, and help you understand complex
                  stories.
                </p>

                {/* Feature Pills */}
                <div className="flex flex-wrap gap-3 justify-center mb-8">
                  <div className="px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
                    🔍 Multi-Source Research
                  </div>
                  <div className="px-4 py-2 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">
                    🎯 Bias Detection
                  </div>
                  <div className="px-4 py-2 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                    ✓ Fact Checking
                  </div>
                  <div className="px-4 py-2 bg-orange-100 text-orange-700 rounded-full text-sm font-medium">
                    📊 Trend Analysis
                  </div>
                </div>

                {/* Example Queries */}
                <div className="w-full max-w-3xl">
                  <p className="text-sm font-semibold text-gray-700 mb-4">
                    Try asking:
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {exampleQueries.map((query, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setInput(query);
                        }}
                        className="p-4 text-left bg-gradient-to-br from-gray-50 to-gray-100 hover:from-blue-50 hover:to-indigo-50 rounded-xl border border-gray-200 hover:border-blue-300 transition-all duration-200 text-sm text-gray-700 hover:text-gray-900 shadow-sm hover:shadow-md"
                      >
                        {query}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              // Messages Display
              messages.map((m: any) => (
                <div
                  key={m.id}
                  className={`flex ${
                    m.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-5 py-4 ${
                      m.role === "user"
                        ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-lg"
                        : "bg-gray-100 text-gray-900 border border-gray-200"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-lg">
                        {m.role === "user" ? "👤" : "📰"}
                      </span>
                      <span className="text-xs font-semibold opacity-90">
                        {m.role === "user" ? "You" : "NewsIQ"}
                      </span>
                    </div>
                    <div className="prose prose-sm max-w-none prose-headings:font-bold prose-h3:text-lg prose-h3:mt-4 prose-h3:mb-2 prose-p:my-2 prose-ul:my-2 prose-li:my-1 prose-a:text-blue-600 prose-a:underline prose-strong:font-semibold">
                      <div
                        className="whitespace-pre-wrap"
                        dangerouslySetInnerHTML={{
                          __html:
                            m.parts
                              ?.map((part: any) => {
                                if (part.type === "text") {
                                  let html = part.text
                                    // Headers
                                    .replace(/### (.*?)$/gm, "<h3>$1</h3>")
                                    // Bold
                                    .replace(
                                      /\*\*(.*?)\*\*/g,
                                      "<strong>$1</strong>"
                                    )
                                    // Links
                                    .replace(
                                      /\[(.*?)\]\((.*?)\)/g,
                                      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
                                    );

                                  html = html.replace(
                                    /(^- .*$\n?)+/gm,
                                    (match: string) => {
                                      const items = match
                                        .split("\n")
                                        .filter((line: string) => line.trim())
                                        .map((line: string) =>
                                          line.replace(/^- /, "")
                                        )
                                        .map((item: any) => `<li>${item}</li>`)
                                        .join("");
                                      return `<ul>${items}</ul>`;
                                    }
                                  );

                                  // Paragraphs
                                  html = html
                                    .split("\n\n")
                                    .map((para: string) => {
                                      if (
                                        para.trim() &&
                                        !para.startsWith("<")
                                      ) {
                                        return `<p>${para}</p>`;
                                      }
                                      return para;
                                    })
                                    .join("");

                                  return html;
                                }
                                return "";
                              })
                              .join("") || "",
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))
            )}

            {/* Loading Indicator */}
            {(status === "submitted" || status === "streaming") && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-2xl px-5 py-4 border border-gray-200">
                  <div className="flex items-center gap-3">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-100"></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-200"></div>
                    </div>
                    <span className="text-sm text-gray-600">
                      Researching news sources...
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 p-5 bg-gray-50">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (input.trim()) {
                  sendMessage({ text: input });
                  setInput("");
                }
              }}
              className="flex gap-3"
            >
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about any news topic, request analysis, or fact-check a claim..."
                className="flex-1 px-5 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white shadow-sm text-gray-900 placeholder-gray-600"
                disabled={status === "submitted" || status === "streaming"}
              />

              <button
                type="submit"
                disabled={
                  status === "submitted" ||
                  status === "streaming" ||
                  !input.trim()
                }
                className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-semibold shadow-lg hover:shadow-xl"
              >
                {status === "submitted" || status === "streaming" ? (
                  <span className="flex items-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Analyzing
                  </span>
                ) : (
                  "Research"
                )}
              </button>
            </form>
            <div className="flex items-center justify-between mt-3">
              <p className="text-xs text-gray-500">
                Powered by Bright Data × Vercel AI SDK
              </p>
              <div className="flex gap-2">
                <span className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-medium">
                  ✓ Real-time
                </span>
                <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                  🌐 Global Sources
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
