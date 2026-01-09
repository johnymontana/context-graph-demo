"use client";

import { useState, useCallback, useEffect } from "react";
import {
  Box,
  Flex,
  Heading,
  Text,
  Container,
  Grid,
  GridItem,
} from "@chakra-ui/react";
import dynamic from "next/dynamic";
import { ChatInterface } from "@/components/ChatInterface";
import { DecisionTracePanel } from "@/components/DecisionTracePanel";
import { getGraphSchema, schemaToGraphData } from "@/lib/api";
import type { Decision, GraphData, GraphNode, ChatMessage } from "@/lib/api";

// Helper to convert a GraphNode to a Decision object
function graphNodeToDecision(node: GraphNode): Decision {
  const props = node.properties;
  return {
    id: (props.id as string) || node.id,
    decision_type: (props.decision_type as string) || "unknown",
    category: (props.category as string) || "unknown",
    reasoning: (props.reasoning as string) || "",
    reasoning_summary: props.reasoning_summary as string | undefined,
    confidence_score: props.confidence_score as number | undefined,
    risk_factors: (props.risk_factors as string[]) || [],
    status: (props.status as string) || "unknown",
    decision_timestamp: props.decision_timestamp as string | undefined,
    timestamp: props.decision_timestamp as string | undefined,
  };
}

// Dynamic import for NVL to avoid SSR issues
const ContextGraphView = dynamic(
  () =>
    import("@/components/ContextGraphView").then((mod) => mod.ContextGraphView),
  { ssr: false },
);

export default function Home() {
  const [selectedDecision, setSelectedDecision] = useState<Decision | null>(
    null,
  );
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [graphDecisions, setGraphDecisions] = useState<Decision[]>([]);
  const [conversationHistory, setConversationHistory] = useState<ChatMessage[]>(
    [],
  );

  // Load graph schema on mount
  useEffect(() => {
    async function loadSchema() {
      try {
        const schema = await getGraphSchema();
        const data = schemaToGraphData(schema);
        setGraphData(data);
      } catch (error) {
        console.error("Failed to load graph schema:", error);
      }
    }
    loadSchema();
  }, []);

  const handleDecisionSelect = useCallback((decision: Decision) => {
    setSelectedDecision(decision);
  }, []);

  const handleGraphUpdate = useCallback((data: GraphData) => {
    setGraphData(data);
  }, []);

  const handleNodeClick = useCallback((nodeId: string, labels: string[]) => {
    console.log("Node clicked:", nodeId, labels);
  }, []);

  // Handle when decision nodes in the graph change
  const handleDecisionNodesChange = useCallback(
    (decisionNodes: GraphNode[]) => {
      const decisions = decisionNodes.map(graphNodeToDecision);
      setGraphDecisions(decisions);
    },
    [],
  );

  // Handle when a decision node is clicked in the graph
  const handleDecisionNodeClick = useCallback((node: GraphNode) => {
    const decision = graphNodeToDecision(node);
    setSelectedDecision(decision);
  }, []);

  return (
    <Box minH="100vh" bg="bg.canvas">
      {/* Header */}
      <Box
        as="header"
        bg="bg.surface"
        borderBottomWidth="1px"
        borderColor="border.default"
        py={4}
        px={6}
      >
        <Container maxW="container.2xl">
          <Flex justify="space-between" align="center">
            <Box>
              <Heading size="lg" color="brand.600">
                Context Graph Demo
              </Heading>
              <Text color="gray.500" fontSize="sm">
                AI-powered decision tracing for financial institutions
              </Text>
            </Box>
          </Flex>
        </Container>
      </Box>

      {/* Main Content */}
      <Container maxW="container.2xl" py={6}>
        <Grid
          templateColumns={{ base: "1fr", lg: "1fr 1fr", xl: "1fr 1.5fr 1fr" }}
          gap={6}
          h="calc(100vh - 140px)"
        >
          {/* Chat Panel */}
          <GridItem overflow="hidden">
            <Box
              bg="bg.surface"
              borderRadius="lg"
              borderWidth="1px"
              borderColor="border.default"
              h="100%"
              display="flex"
              flexDirection="column"
              overflow="hidden"
            >
              <Box
                p={4}
                borderBottomWidth="1px"
                borderColor="border.default"
                flexShrink={0}
              >
                <Heading size="md">AI Assistant</Heading>
                <Text fontSize="sm" color="gray.500">
                  Ask questions about customers, decisions, and policies
                </Text>
              </Box>
              <Box flex="1" minH={0} overflow="hidden">
                <ChatInterface
                  conversationHistory={conversationHistory}
                  onConversationUpdate={setConversationHistory}
                  onDecisionSelect={handleDecisionSelect}
                  onGraphUpdate={handleGraphUpdate}
                />
              </Box>
            </Box>
          </GridItem>

          {/* Graph Visualization */}
          <GridItem overflow="hidden">
            <Box
              bg="bg.surface"
              borderRadius="lg"
              borderWidth="1px"
              borderColor="border.default"
              h="100%"
              display="flex"
              flexDirection="column"
              overflow="hidden"
            >
              <Box
                p={4}
                borderBottomWidth="1px"
                borderColor="border.default"
                flexShrink={0}
              >
                <Heading size="md">Context Graph</Heading>
                <Text fontSize="sm" color="gray.500">
                  Visualize entities, decisions, and causal relationships
                </Text>
              </Box>
              <Box flex="1" minH={0}>
                <ContextGraphView
                  graphData={graphData}
                  onNodeClick={handleNodeClick}
                  onDecisionNodesChange={handleDecisionNodesChange}
                  onDecisionNodeClick={handleDecisionNodeClick}
                  selectedNodeId={selectedDecision?.id}
                />
              </Box>
            </Box>
          </GridItem>

          {/* Decision Trace Panel */}
          <GridItem display={{ base: "none", xl: "block" }} overflow="hidden">
            <Box
              bg="bg.surface"
              borderRadius="lg"
              borderWidth="1px"
              borderColor="border.default"
              h="100%"
              display="flex"
              flexDirection="column"
              overflow="hidden"
            >
              <Box
                p={4}
                borderBottomWidth="1px"
                borderColor="border.default"
                flexShrink={0}
              >
                <Heading size="md">Decision Trace</Heading>
                <Text fontSize="sm" color="gray.500">
                  Inspect reasoning, precedents, and causal chains
                </Text>
              </Box>
              <Box flex="1" minH={0} overflow="auto">
                <DecisionTracePanel
                  decision={selectedDecision}
                  onDecisionSelect={handleDecisionSelect}
                  graphDecisions={graphDecisions}
                />
              </Box>
            </Box>
          </GridItem>
        </Grid>
      </Container>
    </Box>
  );
}
