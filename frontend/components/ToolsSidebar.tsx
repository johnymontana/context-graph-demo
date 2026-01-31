"use client";

import { useState } from "react";
import {
  Box,
  Flex,
  Text,
  VStack,
  HStack,
  Badge,
  Drawer,
  Portal,
  CloseButton,
  Button,
  IconButton,
  Separator,
  Code,
} from "@chakra-ui/react";
import {
  LuWrench,
  LuSearch,
  LuClipboard,
  LuGitBranch,
  LuHistory,
  LuGitMerge,
  LuPlusCircle,
  LuAlertTriangle,
  LuUsers,
  LuFileText,
  LuDatabase,
  LuLayout,
  LuChevronRight,
} from "react-icons/lu";

interface Tool {
  name: string;
  description: string;
  icon: React.ReactNode;
  parameters: string[];
}

const AVAILABLE_TOOLS: Tool[] = [
  {
    name: "search_customer",
    description: "Search for customers by name, email, or account number",
    icon: <LuSearch size={16} />,
    parameters: ["query", "limit"],
  },
  {
    name: "get_customer_decisions",
    description: "Get all decisions made about a specific customer",
    icon: <LuClipboard size={16} />,
    parameters: ["customer_id", "decision_type", "limit"],
  },
  {
    name: "find_similar_decisions",
    description: "Find structurally similar decisions using FastRP embeddings",
    icon: <LuGitBranch size={16} />,
    parameters: ["decision_id", "limit"],
  },
  {
    name: "find_precedents",
    description: "Find precedent decisions using hybrid semantic + structural search",
    icon: <LuHistory size={16} />,
    parameters: ["scenario", "category", "limit"],
  },
  {
    name: "get_causal_chain",
    description: "Trace the causal chain of a decision",
    icon: <LuGitMerge size={16} />,
    parameters: ["decision_id", "direction", "depth"],
  },
  {
    name: "record_decision",
    description: "Record a new decision with full reasoning context",
    icon: <LuPlusCircle size={16} />,
    parameters: [
      "decision_type",
      "category",
      "reasoning",
      "customer_id",
      "confidence_score",
    ],
  },
  {
    name: "detect_fraud_patterns",
    description: "Analyze accounts for potential fraud using graph patterns",
    icon: <LuAlertTriangle size={16} />,
    parameters: ["account_id", "similarity_threshold"],
  },
  {
    name: "find_decision_community",
    description: "Find related decisions using Louvain community detection",
    icon: <LuUsers size={16} />,
    parameters: ["decision_id", "limit"],
  },
  {
    name: "get_policy",
    description: "Get policy rules for a category",
    icon: <LuFileText size={16} />,
    parameters: ["category", "policy_name"],
  },
  {
    name: "execute_cypher",
    description: "Execute a read-only Cypher query",
    icon: <LuDatabase size={16} />,
    parameters: ["cypher"],
  },
  {
    name: "get_schema",
    description: "Get the graph database schema",
    icon: <LuLayout size={16} />,
    parameters: [],
  },
];

interface ToolsSidebarProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onToolSelect?: (toolName: string, prompt: string) => void;
  recentTools?: string[];
}

export function ToolsSidebar({
  open,
  onOpenChange,
  onToolSelect,
  recentTools = [],
}: ToolsSidebarProps) {
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);

  const handleToolClick = (tool: Tool) => {
    if (selectedTool?.name === tool.name) {
      setSelectedTool(null);
    } else {
      setSelectedTool(tool);
    }
  };

  const handleUsePrompt = (tool: Tool) => {
    if (onToolSelect) {
      let prompt = "";
      switch (tool.name) {
        case "search_customer":
          prompt = "Search for customer ";
          break;
        case "get_customer_decisions":
          prompt = "Show me all decisions for customer ";
          break;
        case "find_similar_decisions":
          prompt = "Find decisions similar to ";
          break;
        case "find_precedents":
          prompt = "Find precedents for ";
          break;
        case "get_causal_chain":
          prompt = "Trace the causal chain for decision ";
          break;
        case "detect_fraud_patterns":
          prompt = "Check for fraud patterns on account ";
          break;
        case "get_policy":
          prompt = "What are the policies for ";
          break;
        case "get_schema":
          prompt = "Show me the database schema";
          break;
        default:
          prompt = `Use ${tool.name} to `;
      }
      onToolSelect(tool.name, prompt);
      onOpenChange(false);
    }
  };

  return (
    <Drawer.Root
      open={open}
      onOpenChange={(e) => onOpenChange(e.open)}
      placement="end"
      size="sm"
    >
      <Portal>
        <Drawer.Backdrop />
        <Drawer.Positioner>
          <Drawer.Content>
            <Drawer.Header borderBottomWidth="1px">
              <Drawer.Title>
                <HStack gap={2}>
                  <LuWrench />
                  <Text>Available Tools</Text>
                </HStack>
              </Drawer.Title>
              <Drawer.Description color="text.secondary">
                Tools the AI agent can use to query the context graph
              </Drawer.Description>
              <Drawer.CloseTrigger asChild position="absolute" top={4} right={4}>
                <CloseButton size="sm" />
              </Drawer.CloseTrigger>
            </Drawer.Header>
            <Drawer.Body p={0}>
              {recentTools.length > 0 && (
                <Box p={4} bg="bg.subtle" borderBottomWidth="1px">
                  <Text fontSize="xs" fontWeight="medium" color="text.muted" mb={2}>
                    Recently Used
                  </Text>
                  <HStack gap={2} flexWrap="wrap">
                    {recentTools.slice(0, 3).map((name) => (
                      <Badge key={name} size="sm" colorPalette="blue">
                        {name.replace(/^mcp__graph__/, "").replace(/_/g, " ")}
                      </Badge>
                    ))}
                  </HStack>
                </Box>
              )}

              <VStack align="stretch" gap={0}>
                {AVAILABLE_TOOLS.map((tool) => (
                  <Box key={tool.name}>
                    <Flex
                      px={4}
                      py={3}
                      cursor="pointer"
                      _hover={{ bg: "bg.subtle" }}
                      onClick={() => handleToolClick(tool)}
                      justify="space-between"
                      align="center"
                      bg={selectedTool?.name === tool.name ? "bg.subtle" : "transparent"}
                    >
                      <HStack gap={3}>
                        <Box color="text.secondary">{tool.icon}</Box>
                        <Box>
                          <Text fontSize="sm" fontWeight="medium">
                            {tool.name.replace(/_/g, " ")}
                          </Text>
                          <Text fontSize="xs" color="text.muted" lineClamp={1}>
                            {tool.description}
                          </Text>
                        </Box>
                      </HStack>
                      <LuChevronRight
                        size={16}
                        style={{
                          transform:
                            selectedTool?.name === tool.name
                              ? "rotate(90deg)"
                              : "rotate(0deg)",
                          transition: "transform 0.2s",
                        }}
                      />
                    </Flex>

                    {selectedTool?.name === tool.name && (
                      <Box px={4} pb={3} bg="bg.subtle">
                        <Text fontSize="xs" color="text.secondary" mb={2}>
                          {tool.description}
                        </Text>
                        {tool.parameters.length > 0 && (
                          <Box mb={3}>
                            <Text fontSize="xs" color="text.muted" mb={1}>
                              Parameters:
                            </Text>
                            <HStack gap={1} flexWrap="wrap">
                              {tool.parameters.map((param) => (
                                <Code key={param} fontSize="xs" px={1}>
                                  {param}
                                </Code>
                              ))}
                            </HStack>
                          </Box>
                        )}
                        <Button
                          size="xs"
                          colorPalette="blue"
                          onClick={() => handleUsePrompt(tool)}
                        >
                          Use this tool
                        </Button>
                      </Box>
                    )}
                    <Separator />
                  </Box>
                ))}
              </VStack>
            </Drawer.Body>
          </Drawer.Content>
        </Drawer.Positioner>
      </Portal>
    </Drawer.Root>
  );
}

export function ToolsButton({
  onClick,
  recentCount = 0,
}: {
  onClick: () => void;
  recentCount?: number;
}) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      position="relative"
    >
      <LuWrench />
      <Text display={{ base: "none", sm: "inline" }}>Tools</Text>
      {recentCount > 0 && (
        <Badge
          colorPalette="blue"
          size="xs"
          position="absolute"
          top={-1}
          right={-1}
          borderRadius="full"
        >
          {recentCount}
        </Badge>
      )}
    </Button>
  );
}
