import type { Metadata, Viewport } from "next";
import { Provider } from "@/components/ui/provider";
import { Analytics } from "@vercel/analytics/next";

export const metadata: Metadata = {
  title: "Lenny's Memory | Neo4j Agent Memory Demo",
  description:
    "Build context graphs for AI agents with Neo4j Agent Memory. An interactive demo showcasing decision tracing, semantic search, and graph-powered agent memory.",
  keywords: [
    "Neo4j",
    "Agent Memory",
    "AI",
    "Context Graph",
    "Knowledge Graph",
    "LLM",
    "RAG",
  ],
  authors: [{ name: "Neo4j Labs" }],
  openGraph: {
    title: "Lenny's Memory | Neo4j Agent Memory Demo",
    description:
      "Build context graphs for AI agents with Neo4j Agent Memory. Interactive demo showcasing decision tracing and graph-powered memory.",
    type: "website",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0a" },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Public+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <Provider>{children}</Provider>
        <Analytics />
      </body>
    </html>
  );
}
