"use client";

import { useEffect, useCallback, useMemo, useState, useRef } from "react";
import {
  Box,
  Text,
  Flex,
  Badge,
  VStack,
  HStack,
  Heading,
  CloseButton,
  Spinner,
} from "@chakra-ui/react";
import type { GraphData, GraphNode, GraphRelationship } from "@/lib/api";
import { expandNode, getRelationshipsBetween } from "@/lib/api";

// NVL types
interface NvlNode {
  id: string;
  caption?: string;
  color?: string;
  size?: number;
  selected?: boolean;
}

interface NvlRelationship {
  id: string;
  from: string;
  to: string;
  caption?: string;
  color?: string;
  selected?: boolean;
}

// Node color mapping by label
const NODE_COLORS: Record<string, string> = {
  Person: "#4299E1",
  Account: "#48BB78",
  Transaction: "#ED8936",
  Decision: "#9F7AEA",
  Organization: "#F56565",
  Policy: "#38B2AC",
  Exception: "#D69E2E",
  Escalation: "#805AD5",
  Employee: "#63B3ED",
  DecisionContext: "#B794F4",
  Precedent: "#F687B3",
  RelationshipType: "#A0AEC0",
  Community: "#DD6B20",
  SupportTicket: "#3182CE",
  Alert: "#ECC94B",
};

// Node size by label
const NODE_SIZES: Record<string, number> = {
  Decision: 30,
  Person: 25,
  Account: 22,
  Transaction: 18,
  Organization: 25,
  Policy: 22,
  Exception: 20,
  Escalation: 20,
  Employee: 22,
  DecisionContext: 18,
  Precedent: 18,
  RelationshipType: 20,
  Community: 28,
  SupportTicket: 20,
  Alert: 22,
};

interface SelectedElement {
  type: "node" | "relationship";
  data: GraphNode | GraphRelationship;
}

interface ContextGraphViewProps {
  graphData: GraphData | null;
  onNodeClick?: (nodeId: string, labels: string[]) => void;
  onGraphDataChange?: (graphData: GraphData) => void;
  onDecisionNodesChange?: (decisions: GraphNode[]) => void;
  onDecisionNodeClick?: (node: GraphNode) => void;
  selectedNodeId?: string;
  height?: string;
  showLegend?: boolean;
}

export function ContextGraphView({
  graphData,
  onNodeClick,
  onGraphDataChange,
  onDecisionNodesChange,
  onDecisionNodeClick,
  selectedNodeId,
  height = "100%",
  showLegend = true,
}: ContextGraphViewProps) {
  const [selectedElement, setSelectedElement] =
    useState<SelectedElement | null>(null);
  const [internalSelectedNodeId, setInternalSelectedNodeId] = useState<
    string | null
  >(selectedNodeId || null);
  const [internalSelectedRelId, setInternalSelectedRelId] = useState<
    string | null
  >(null);
  const [isExpanding, setIsExpanding] = useState(false);
  const [expandedNodeIds, setExpandedNodeIds] = useState<Set<string>>(
    new Set(),
  );

  // Track internal graph data for expansions
  const [internalGraphData, setInternalGraphData] = useState<GraphData | null>(
    graphData,
  );

  // Update internal graph data when prop changes
  useEffect(() => {
    if (graphData) {
      setInternalGraphData(graphData);
      // Reset expanded nodes when new graph data comes in
      setExpandedNodeIds(new Set());
    }
  }, [graphData]);

  // Notify parent when decision nodes change (excluding schema nodes)
  useEffect(() => {
    if (internalGraphData && onDecisionNodesChange) {
      const decisionNodes = internalGraphData.nodes.filter(
        (node) =>
          node.labels.includes("Decision") && !node.properties.isSchemaNode,
      );
      onDecisionNodesChange(decisionNodes);
    }
  }, [internalGraphData, onDecisionNodesChange]);

  // Transform graph data to NVL format
  const nvlData = useMemo(() => {
    if (!internalGraphData) return { nodes: [], relationships: [] };

    const nodes: NvlNode[] = internalGraphData.nodes.map((node) => {
      const primaryLabel = node.labels[0] || "Unknown";
      const isSchemaNode = node.properties.isSchemaNode as boolean;
      const count = node.properties.count as number;
      const isExpanded = expandedNodeIds.has(node.id);

      let caption =
        (node.properties.name as string) ||
        (node.properties.first_name as string) ||
        (node.properties.decision_type as string) ||
        node.id.slice(0, 8);

      // Add count to caption for schema nodes
      if (isSchemaNode && count !== undefined) {
        caption = `${caption} (${count})`;
      }

      const isSelected = internalSelectedNodeId === node.id;

      return {
        id: node.id,
        caption,
        color: isSelected
          ? "#E53E3E"
          : isExpanded
            ? "#38A169" // Green for expanded nodes
            : NODE_COLORS[primaryLabel] || "#718096",
        size: isSelected
          ? (NODE_SIZES[primaryLabel] || 20) * 1.3
          : NODE_SIZES[primaryLabel] || 20,
        selected: isSelected,
      };
    });

    const relationships: NvlRelationship[] =
      internalGraphData.relationships.map((rel) => {
        const isSelected = internalSelectedRelId === rel.id;
        return {
          id: rel.id,
          from: rel.startNodeId,
          to: rel.endNodeId,
          caption: rel.type,
          color: isSelected
            ? "#E53E3E"
            : rel.type === "CAUSED"
              ? "#E53E3E"
              : rel.type === "INFLUENCED"
                ? "#D69E2E"
                : "#A0AEC0",
          selected: isSelected,
        };
      });

    return { nodes, relationships };
  }, [
    internalGraphData,
    internalSelectedNodeId,
    internalSelectedRelId,
    expandedNodeIds,
  ]);

  const handleNodeClick = useCallback(
    (node: NvlNode) => {
      if (internalGraphData) {
        const originalNode = internalGraphData.nodes.find(
          (n) => n.id === node.id,
        );
        if (originalNode) {
          setSelectedElement({ type: "node", data: originalNode });
          setInternalSelectedNodeId(node.id);
          setInternalSelectedRelId(null);
          if (onNodeClick) {
            onNodeClick(node.id, originalNode.labels);
          }
          // If it's a Decision node, notify the parent
          if (originalNode.labels.includes("Decision") && onDecisionNodeClick) {
            onDecisionNodeClick(originalNode);
          }
        }
      }
    },
    [internalGraphData, onNodeClick, onDecisionNodeClick],
  );

  // Handle double-click to expand node
  const handleNodeDoubleClick = useCallback(
    async (node: NvlNode) => {
      if (!internalGraphData || isExpanding) return;

      // Don't expand if already expanded
      if (expandedNodeIds.has(node.id)) return;

      setIsExpanding(true);

      try {
        // Fetch connected nodes
        const expandedData = await expandNode(node.id);

        if (expandedData.nodes.length === 0) {
          setIsExpanding(false);
          return;
        }

        // Merge new nodes with existing ones (avoid duplicates)
        const existingNodeIds = new Set(
          internalGraphData.nodes.map((n) => n.id),
        );
        const newNodes = expandedData.nodes.filter(
          (n) => !existingNodeIds.has(n.id),
        );

        // Merge new relationships (avoid duplicates)
        const existingRelIds = new Set(
          internalGraphData.relationships.map((r) => r.id),
        );
        const newRels = expandedData.relationships.filter(
          (r) => !existingRelIds.has(r.id),
        );

        // Create updated graph data
        const updatedNodes = [...internalGraphData.nodes, ...newNodes];
        const updatedRels = [...internalGraphData.relationships, ...newRels];

        // Fetch any missing relationships between all nodes
        const allNodeIds = updatedNodes.map((n) => n.id);
        const additionalRels = await getRelationshipsBetween(allNodeIds);

        // Filter out already existing relationships
        const allExistingRelIds = new Set(updatedRels.map((r) => r.id));
        const trulyNewRels = additionalRels.filter(
          (r) => !allExistingRelIds.has(r.id),
        );

        const finalGraphData: GraphData = {
          nodes: updatedNodes,
          relationships: [...updatedRels, ...trulyNewRels],
        };

        // Update internal state
        setInternalGraphData(finalGraphData);
        setExpandedNodeIds((prev) => {
          const newSet = new Set(prev);
          newSet.add(node.id);
          return newSet;
        });

        // Notify parent if callback provided
        if (onGraphDataChange) {
          onGraphDataChange(finalGraphData);
        }
      } catch (error) {
        console.error("Error expanding node:", error);
      } finally {
        setIsExpanding(false);
      }
    },
    [internalGraphData, isExpanding, expandedNodeIds, onGraphDataChange],
  );

  const handleRelationshipClick = useCallback(
    (rel: NvlRelationship) => {
      if (internalGraphData) {
        const originalRel = internalGraphData.relationships.find(
          (r) => r.id === rel.id,
        );
        if (originalRel) {
          setSelectedElement({ type: "relationship", data: originalRel });
          setInternalSelectedRelId(rel.id);
          setInternalSelectedNodeId(null);
        }
      }
    },
    [internalGraphData],
  );

  const handleCanvasClick = useCallback(() => {
    // Deselect when clicking on empty canvas
    setSelectedElement(null);
    setInternalSelectedNodeId(null);
    setInternalSelectedRelId(null);
  }, []);

  const handleClosePanel = useCallback(() => {
    setSelectedElement(null);
    setInternalSelectedNodeId(null);
    setInternalSelectedRelId(null);
  }, []);

  if (!internalGraphData || internalGraphData.nodes.length === 0) {
    return (
      <Flex
        h={height}
        align="center"
        justify="center"
        direction="column"
        gap={4}
        p={8}
      >
        <Text color="gray.500" textAlign="center">
          No graph data to display.
        </Text>
        <Text color="gray.400" fontSize="sm" textAlign="center">
          Use the AI assistant to search for customers or decisions to visualize
          the context graph.
        </Text>
      </Flex>
    );
  }

  return (
    <Box h={height} position="relative">
      {/* Legend */}
      {showLegend && (
        <Flex
          position="absolute"
          top={2}
          left={2}
          zIndex={10}
          bg="bg.surface"
          borderRadius="md"
          p={2}
          gap={2}
          flexWrap="wrap"
          maxW="200px"
          boxShadow="sm"
          borderWidth="1px"
          borderColor="border.default"
        >
          {Object.entries(NODE_COLORS)
            .slice(0, 6)
            .map(([label, color]) => (
              <Badge
                key={label}
                size="sm"
                style={{ backgroundColor: color, color: "white" }}
              >
                {label}
              </Badge>
            ))}
        </Flex>
      )}

      {/* Properties Panel */}
      {selectedElement && (
        <Box
          position="absolute"
          top={2}
          right={2}
          zIndex={10}
          bg="bg.surface"
          borderRadius="md"
          p={3}
          maxW="280px"
          maxH="400px"
          overflow="auto"
          boxShadow="md"
          borderWidth="1px"
          borderColor="border.default"
        >
          <Flex justify="space-between" align="center" mb={2}>
            <Heading size="sm">
              {selectedElement.type === "node" ? "Node" : "Relationship"}{" "}
              Properties
            </Heading>
            <CloseButton size="sm" onClick={handleClosePanel} />
          </Flex>

          {selectedElement.type === "node" && (
            <VStack align="stretch" gap={2}>
              <HStack>
                <Text fontSize="xs" fontWeight="bold" color="gray.500">
                  Labels:
                </Text>
                <Flex gap={1} flexWrap="wrap">
                  {(selectedElement.data as GraphNode).labels.map((label) => (
                    <Badge
                      key={label}
                      size="sm"
                      style={{
                        backgroundColor: NODE_COLORS[label] || "#718096",
                        color: "white",
                      }}
                    >
                      {label}
                    </Badge>
                  ))}
                </Flex>
              </HStack>
              <Box>
                <Text fontSize="xs" fontWeight="bold" color="gray.500" mb={1}>
                  Properties:
                </Text>
                <VStack align="stretch" gap={1}>
                  {Object.entries(
                    (selectedElement.data as GraphNode).properties,
                  )
                    .filter(([key]) => key !== "fast_rp_embedding")
                    .map(([key, value]) => (
                    <Box
                      key={key}
                      bg="bg.subtle"
                      p={1}
                      borderRadius="sm"
                      fontSize="xs"
                    >
                      <Text fontWeight="medium" color="gray.600">
                        {key}:
                      </Text>
                      <Text
                        color="gray.800"
                        wordBreak="break-word"
                        whiteSpace="pre-wrap"
                      >
                        {typeof value === "object"
                          ? JSON.stringify(value, null, 2)
                          : String(value)}
                      </Text>
                    </Box>
                  ))}
                </VStack>
              </Box>
            </VStack>
          )}

          {selectedElement.type === "relationship" && (
            <VStack align="stretch" gap={2}>
              <HStack>
                <Text fontSize="xs" fontWeight="bold" color="gray.500">
                  Type:
                </Text>
                <Badge size="sm" colorPalette="gray">
                  {(selectedElement.data as GraphRelationship).type}
                </Badge>
              </HStack>
              <Box>
                <Text fontSize="xs" fontWeight="bold" color="gray.500">
                  From:{" "}
                  <Text as="span" fontWeight="normal">
                    {(selectedElement.data as GraphRelationship).startNodeId}
                  </Text>
                </Text>
              </Box>
              <Box>
                <Text fontSize="xs" fontWeight="bold" color="gray.500">
                  To:{" "}
                  <Text as="span" fontWeight="normal">
                    {(selectedElement.data as GraphRelationship).endNodeId}
                  </Text>
                </Text>
              </Box>
              {Object.keys(
                (selectedElement.data as GraphRelationship).properties,
              ).length > 0 && (
                <Box>
                  <Text fontSize="xs" fontWeight="bold" color="gray.500" mb={1}>
                    Properties:
                  </Text>
                  <VStack align="stretch" gap={1}>
                    {Object.entries(
                      (selectedElement.data as GraphRelationship).properties,
                    ).map(([key, value]) => (
                      <Box
                        key={key}
                        bg="bg.subtle"
                        p={1}
                        borderRadius="sm"
                        fontSize="xs"
                      >
                        <Text fontWeight="medium" color="gray.600">
                          {key}:
                        </Text>
                        <Text color="gray.800" wordBreak="break-word">
                          {typeof value === "object"
                            ? JSON.stringify(value, null, 2)
                            : String(value)}
                        </Text>
                      </Box>
                    ))}
                  </VStack>
                </Box>
              )}
            </VStack>
          )}
        </Box>
      )}

      {/* Instructions */}
      <Box
        position="absolute"
        bottom={2}
        left={2}
        zIndex={10}
        bg="bg.surface"
        borderRadius="md"
        px={2}
        py={1}
        boxShadow="sm"
        borderWidth="1px"
        borderColor="border.default"
        opacity={0.8}
      >
        <Text fontSize="xs" color="gray.500">
          Scroll to zoom | Drag canvas to pan | Drag nodes to move | Click to
          inspect | Double-click to expand
        </Text>
      </Box>

      {/* Loading indicator for expansion */}
      {isExpanding && (
        <Flex
          position="absolute"
          top="50%"
          left="50%"
          transform="translate(-50%, -50%)"
          zIndex={20}
          bg="bg.surface"
          borderRadius="md"
          p={3}
          boxShadow="md"
          borderWidth="1px"
          borderColor="border.default"
          align="center"
          gap={2}
        >
          <Spinner size="sm" />
          <Text fontSize="sm">Expanding node...</Text>
        </Flex>
      )}

      {/* Graph Container */}
      <Box h="100%" w="100%">
        <NvlGraph
          nodes={nvlData.nodes}
          relationships={nvlData.relationships}
          onNodeClick={handleNodeClick}
          onNodeDoubleClick={handleNodeDoubleClick}
          onRelationshipClick={handleRelationshipClick}
          onCanvasClick={handleCanvasClick}
        />
      </Box>
    </Box>
  );
}

// Separate component for NVL to handle dynamic import properly
function NvlGraph({
  nodes,
  relationships,
  onNodeClick,
  onNodeDoubleClick,
  onRelationshipClick,
  onCanvasClick,
}: {
  nodes: NvlNode[];
  relationships: NvlRelationship[];
  onNodeClick: (node: NvlNode) => void;
  onNodeDoubleClick: (node: NvlNode) => void;
  onRelationshipClick: (rel: NvlRelationship) => void;
  onCanvasClick: () => void;
}) {
  const [NvlComponent, setNvlComponent] =
    useState<React.ComponentType<any> | null>(null);
  const [isReady, setIsReady] = useState(false);
  const nvlRef = useRef<any>(null);

  useEffect(() => {
    import("@neo4j-nvl/react").then((mod) => {
      setNvlComponent(() => mod.InteractiveNvlWrapper);
    });
  }, []);

  // Mark as ready after a short delay to allow NVL instance to fully initialize
  useEffect(() => {
    if (NvlComponent && nodes.length > 0) {
      const timer = setTimeout(() => {
        setIsReady(true);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [NvlComponent, nodes.length]);

  if (!NvlComponent) {
    return (
      <Flex h="100%" align="center" justify="center">
        <Text color="gray.500">Loading graph visualization...</Text>
      </Flex>
    );
  }

  return (
    <NvlComponent
      ref={nvlRef}
      nodes={nodes}
      rels={relationships}
      nvlOptions={{
        layout: "d3Force",
        initialZoom: 1,
        minZoom: 0.1,
        maxZoom: 5,
        relationshipThickness: 2,
        disableTelemetry: true,
      }}
      mouseEventCallbacks={{
        onNodeClick: (node: NvlNode) => onNodeClick(node),
        onNodeDoubleClick: (node: NvlNode) => onNodeDoubleClick(node),
        onRelationshipClick: (rel: NvlRelationship) => onRelationshipClick(rel),
        onCanvasClick: () => onCanvasClick(),
        onZoom: isReady,
        onPan: isReady,
        onDrag: isReady,
      }}
      style={{ width: "100%", height: "100%" }}
    />
  );
}

// Compact inline graph for chat messages
interface InlineGraphProps {
  graphData: GraphData;
  height?: string;
  onNodeClick?: (nodeId: string, labels: string[]) => void;
}

export function InlineGraph({
  graphData,
  height = "200px",
  onNodeClick,
}: InlineGraphProps) {
  return (
    <Box
      borderRadius="md"
      borderWidth="1px"
      borderColor="border.default"
      overflow="hidden"
      h={height}
      my={2}
    >
      <ContextGraphView
        graphData={graphData}
        onNodeClick={onNodeClick}
        showLegend={false}
        height={height}
      />
    </Box>
  );
}
