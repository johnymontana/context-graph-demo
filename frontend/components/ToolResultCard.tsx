"use client";

import { useState, useMemo } from "react";
import {
  Box,
  Flex,
  Text,
  VStack,
  HStack,
  Badge,
  Code,
  IconButton,
  Tabs,
} from "@chakra-ui/react";
import { LuChevronDown, LuChevronUp, LuCopy, LuCheck } from "react-icons/lu";
import dynamic from "next/dynamic";
import type { GraphData } from "@/lib/api";

const ContextGraphView = dynamic(
  () =>
    import("@/components/ContextGraphView").then((mod) => mod.ContextGraphView),
  { ssr: false }
);

interface ToolResultCardProps {
  toolName: string;
  input: Record<string, unknown>;
  output: unknown;
  graphData?: GraphData;
  isLoading?: boolean;
}

const TOOL_ICONS: Record<string, string> = {
  search_customer: "search",
  get_customer_decisions: "clipboard",
  find_similar_decisions: "git-branch",
  find_precedents: "history",
  get_causal_chain: "git-merge",
  record_decision: "plus-circle",
  detect_fraud_patterns: "alert-triangle",
  find_decision_community: "users",
  get_policy: "file-text",
  execute_cypher: "database",
  get_schema: "layout",
};

const TOOL_COLORS: Record<string, string> = {
  search_customer: "blue",
  get_customer_decisions: "purple",
  find_similar_decisions: "cyan",
  find_precedents: "orange",
  get_causal_chain: "green",
  record_decision: "teal",
  detect_fraud_patterns: "red",
  find_decision_community: "pink",
  get_policy: "gray",
  execute_cypher: "yellow",
  get_schema: "gray",
};

const TOOL_DESCRIPTIONS: Record<string, string> = {
  search_customer: "Customer search results",
  get_customer_decisions: "Decisions for this customer",
  find_similar_decisions: "Structurally similar decisions",
  find_precedents: "Relevant precedents found",
  get_causal_chain: "Causal relationships traced",
  record_decision: "Decision recorded",
  detect_fraud_patterns: "Fraud pattern analysis",
  find_decision_community: "Related decision community",
  get_policy: "Applicable policies",
  execute_cypher: "Query results",
  get_schema: "Database schema",
};

function formatToolName(name: string): string {
  return name
    .replace(/^mcp__graph__/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function extractSummary(
  toolName: string,
  output: unknown
): { primary: string; secondary: string; stats: string[] } {
  const cleanName = toolName.replace(/^mcp__graph__/, "");
  const result: { primary: string; secondary: string; stats: string[] } = {
    primary: "",
    secondary: "",
    stats: [],
  };

  if (!output || typeof output !== "object") {
    return result;
  }

  const data = output as Record<string, unknown>;

  switch (cleanName) {
    case "search_customer":
      if (Array.isArray(data.customers)) {
        const count = data.customers.length;
        result.primary = `Found ${count} customer${count !== 1 ? "s" : ""}`;
        if (count > 0) {
          const first = data.customers[0] as Record<string, unknown>;
          result.secondary = `${first.first_name || ""} ${first.last_name || ""}`.trim();
          if (first.risk_score !== undefined) {
            result.stats.push(`Risk: ${((first.risk_score as number) * 100).toFixed(0)}%`);
          }
        }
      }
      break;

    case "find_precedents":
      if (Array.isArray(data.precedents)) {
        const count = data.precedents.length;
        result.primary = `${count} precedent${count !== 1 ? "s" : ""} found`;
        if (count > 0) {
          const scores = data.precedents
            .map((p: Record<string, unknown>) => p.combined_score as number)
            .filter(Boolean);
          if (scores.length > 0) {
            const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
            result.stats.push(`Avg similarity: ${(avgScore * 100).toFixed(0)}%`);
          }
        }
      }
      break;

    case "find_similar_decisions":
      if (Array.isArray(data.similar_decisions)) {
        const count = data.similar_decisions.length;
        result.primary = `${count} similar decision${count !== 1 ? "s" : ""}`;
        if (count > 0) {
          const first = data.similar_decisions[0] as Record<string, unknown>;
          if (first.similarity !== undefined) {
            result.stats.push(`Top match: ${((first.similarity as number) * 100).toFixed(0)}%`);
          }
        }
      }
      break;

    case "get_causal_chain":
      if (data.causal_chain && typeof data.causal_chain === "object") {
        const chain = data.causal_chain as Record<string, unknown>;
        const causes = Array.isArray(chain.causes) ? chain.causes.length : 0;
        const effects = Array.isArray(chain.effects) ? chain.effects.length : 0;
        result.primary = "Causal chain traced";
        result.stats.push(`${causes} cause${causes !== 1 ? "s" : ""}`);
        result.stats.push(`${effects} effect${effects !== 1 ? "s" : ""}`);
      }
      break;

    case "get_customer_decisions":
      if (Array.isArray(data.decisions)) {
        const count = data.decisions.length;
        result.primary = `${count} decision${count !== 1 ? "s" : ""} found`;
        const types = [...new Set(data.decisions.map((d: Record<string, unknown>) => d.decision_type))];
        if (types.length > 0) {
          result.secondary = types.slice(0, 3).join(", ");
        }
      }
      break;

    case "detect_fraud_patterns":
      if (Array.isArray(data)) {
        result.primary = data.length > 0 ? "Patterns detected" : "No patterns found";
        result.stats.push(`${data.length} match${data.length !== 1 ? "es" : ""}`);
      }
      break;

    case "get_policy":
      if (Array.isArray(data.matching_policies)) {
        const count = data.matching_policies.length;
        result.primary = `${count} polic${count !== 1 ? "ies" : "y"} found`;
      } else if (Array.isArray(data)) {
        result.primary = `${data.length} polic${data.length !== 1 ? "ies" : "y"} found`;
      }
      break;

    case "find_decision_community":
      if (Array.isArray(data.community_decisions)) {
        const count = data.community_decisions.length;
        result.primary = `${count} related decision${count !== 1 ? "s" : ""}`;
      }
      break;

    case "get_schema":
      result.primary = "Schema retrieved";
      if (Array.isArray(data.node_labels)) {
        result.stats.push(`${data.node_labels.length} node types`);
      }
      if (Array.isArray(data.relationship_types)) {
        result.stats.push(`${data.relationship_types.length} rel types`);
      }
      break;

    default:
      result.primary = TOOL_DESCRIPTIONS[cleanName] || "Results returned";
  }

  return result;
}

function formatJSON(obj: unknown): string {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

export function ToolResultCard({
  toolName,
  input,
  output,
  graphData,
  isLoading,
}: ToolResultCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<string>("graph");

  const cleanToolName = toolName.replace(/^mcp__graph__/, "");
  const colorScheme = TOOL_COLORS[cleanToolName] || "gray";
  const summary = useMemo(
    () => extractSummary(toolName, output),
    [toolName, output]
  );

  const hasGraph = graphData && graphData.nodes && graphData.nodes.length > 0;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(formatJSON(output));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Box
      bg="bg.surface"
      borderRadius="xl"
      borderWidth="1px"
      borderColor="border.default"
      overflow="hidden"
      shadow="sm"
      _hover={{ shadow: "md" }}
      transition="shadow 0.2s"
    >
      <Flex
        px={4}
        py={3}
        bg="bg.subtle"
        justify="space-between"
        align="center"
        cursor="pointer"
        onClick={() => setIsExpanded(!isExpanded)}
        borderBottomWidth={isExpanded ? "1px" : "0"}
        borderColor="border.default"
      >
        <HStack gap={3}>
          <Badge
            colorPalette={colorScheme}
            size="sm"
            px={2}
            py={1}
            borderRadius="md"
          >
            {formatToolName(toolName)}
          </Badge>
          {!isLoading && summary.primary && (
            <Text fontSize="sm" fontWeight="medium" color="text.primary">
              {summary.primary}
            </Text>
          )}
          {!isLoading && summary.stats.length > 0 && (
            <HStack gap={2} display={{ base: "none", sm: "flex" }}>
              {summary.stats.map((stat, i) => (
                <Badge key={i} variant="subtle" size="sm" colorPalette="gray">
                  {stat}
                </Badge>
              ))}
            </HStack>
          )}
          {isLoading && (
            <Text fontSize="sm" color="text.muted">
              Running...
            </Text>
          )}
        </HStack>
        <IconButton
          aria-label={isExpanded ? "Collapse" : "Expand"}
          size="sm"
          variant="ghost"
        >
          {isExpanded ? <LuChevronUp /> : <LuChevronDown />}
        </IconButton>
      </Flex>

      {isExpanded && (
        <Box>
          <Tabs.Root
            value={activeTab}
            onValueChange={(e) => setActiveTab(e.value)}
            size="sm"
          >
            <Tabs.List px={4} pt={2} borderBottomWidth="1px" borderColor="border.subtle">
              {hasGraph && (
                <Tabs.Trigger value="graph" px={3} py={2}>
                  Graph
                </Tabs.Trigger>
              )}
              <Tabs.Trigger value="summary" px={3} py={2}>
                Summary
              </Tabs.Trigger>
              <Tabs.Trigger value="input" px={3} py={2}>
                Input
              </Tabs.Trigger>
              <Tabs.Trigger value="output" px={3} py={2}>
                Output
              </Tabs.Trigger>
            </Tabs.List>

            {hasGraph && (
              <Tabs.Content value="graph" p={0}>
                <Box h={{ base: "250px", md: "300px" }} position="relative">
                  <ContextGraphView
                    graphData={graphData}
                    height="100%"
                    showLegend={true}
                  />
                  <Box
                    position="absolute"
                    bottom={2}
                    right={2}
                    bg="bg.surface"
                    px={2}
                    py={1}
                    borderRadius="md"
                    fontSize="xs"
                    color="text.muted"
                    borderWidth="1px"
                    borderColor="border.subtle"
                  >
                    {graphData.nodes.length} nodes &bull;{" "}
                    {graphData.relationships.length} relationships
                  </Box>
                </Box>
              </Tabs.Content>
            )}

            <Tabs.Content value="summary" p={4}>
              <VStack align="stretch" gap={3}>
                {summary.secondary && (
                  <Box>
                    <Text fontSize="xs" color="text.muted" mb={1}>
                      Details
                    </Text>
                    <Text fontSize="sm">{summary.secondary}</Text>
                  </Box>
                )}
                {summary.stats.length > 0 && (
                  <Box>
                    <Text fontSize="xs" color="text.muted" mb={1}>
                      Statistics
                    </Text>
                    <HStack gap={2} flexWrap="wrap">
                      {summary.stats.map((stat, i) => (
                        <Badge key={i} variant="outline" size="sm">
                          {stat}
                        </Badge>
                      ))}
                    </HStack>
                  </Box>
                )}
                {!summary.secondary && summary.stats.length === 0 && (
                  <Text fontSize="sm" color="text.muted">
                    View the Output tab for full results.
                  </Text>
                )}
              </VStack>
            </Tabs.Content>

            <Tabs.Content value="input" p={4}>
              <Code
                display="block"
                whiteSpace="pre-wrap"
                p={3}
                borderRadius="md"
                fontSize="xs"
                bg="bg.subtle"
                maxH="200px"
                overflowY="auto"
              >
                {formatJSON(input)}
              </Code>
            </Tabs.Content>

            <Tabs.Content value="output" p={4}>
              <Flex justify="flex-end" mb={2}>
                <IconButton
                  aria-label="Copy output"
                  size="xs"
                  variant="ghost"
                  onClick={handleCopy}
                >
                  {copied ? <LuCheck /> : <LuCopy />}
                </IconButton>
              </Flex>
              <Code
                display="block"
                whiteSpace="pre-wrap"
                p={3}
                borderRadius="md"
                fontSize="xs"
                bg="bg.subtle"
                maxH="300px"
                overflowY="auto"
              >
                {formatJSON(output)}
              </Code>
            </Tabs.Content>
          </Tabs.Root>
        </Box>
      )}

      {!isExpanded && hasGraph && (
        <Box h="150px" position="relative">
          <ContextGraphView graphData={graphData} height="100%" showLegend={false} />
          <Box
            position="absolute"
            bottom={2}
            right={2}
            bg="bg.surface"
            px={2}
            py={1}
            borderRadius="md"
            fontSize="xs"
            color="text.muted"
            opacity={0.9}
          >
            {graphData.nodes.length} nodes
          </Box>
        </Box>
      )}
    </Box>
  );
}

export function ToolResultCardCompact({
  toolName,
  isLoading,
}: {
  toolName: string;
  isLoading?: boolean;
}) {
  const cleanToolName = toolName.replace(/^mcp__graph__/, "");
  const colorScheme = TOOL_COLORS[cleanToolName] || "gray";

  return (
    <Flex
      px={3}
      py={2}
      bg="bg.subtle"
      borderRadius="lg"
      align="center"
      gap={2}
      borderWidth="1px"
      borderColor="border.default"
    >
      <Badge colorPalette={colorScheme} size="sm">
        {formatToolName(toolName)}
      </Badge>
      <Text fontSize="sm" color="text.muted">
        {isLoading ? "Running..." : "Completed"}
      </Text>
    </Flex>
  );
}
