"use client";

import { useState, useCallback } from "react";
import {
  Box,
  Flex,
  Heading,
  Text,
  Container,
  Button,
  Menu,
  Portal,
} from "@chakra-ui/react";
import { LuMenu, LuDatabase } from "react-icons/lu";
import { ChatInterface } from "@/components/ChatInterface";
import { SchemaDrawer } from "@/components/SchemaDrawer";
import { ToolsSidebar, ToolsButton } from "@/components/ToolsSidebar";
import type { Decision, GraphData, ChatMessage } from "@/lib/api";

export default function Home() {
  const [selectedDecision, setSelectedDecision] = useState<Decision | null>(
    null
  );
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [conversationHistory, setConversationHistory] = useState<ChatMessage[]>(
    []
  );
  const [schemaDrawerOpen, setSchemaDrawerOpen] = useState(false);
  const [toolsSidebarOpen, setToolsSidebarOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");

  const handleDecisionSelect = useCallback((decision: Decision) => {
    setSelectedDecision(decision);
  }, []);

  const handleGraphUpdate = useCallback((data: GraphData) => {
    setGraphData(data);
  }, []);

  const handleToolSelect = useCallback((toolName: string, prompt: string) => {
    setInputValue(prompt);
  }, []);

  const recentTools = conversationHistory
    .flatMap((msg) => {
      if (msg.role === "assistant" && "toolCalls" in msg) {
        const toolCalls = (msg as { toolCalls?: { name: string }[] }).toolCalls;
        return toolCalls?.map((t) => t.name) || [];
      }
      return [];
    })
    .slice(-5);

  return (
    <Box minH="100vh" bg="bg.canvas">
      <SchemaDrawer open={schemaDrawerOpen} onOpenChange={setSchemaDrawerOpen} />
      <ToolsSidebar
        open={toolsSidebarOpen}
        onOpenChange={setToolsSidebarOpen}
        onToolSelect={handleToolSelect}
        recentTools={recentTools}
      />

      <Box
        as="header"
        bg="bg.surface"
        borderBottomWidth="1px"
        borderColor="border.default"
        py={{ base: 2, md: 3 }}
        px={{ base: 3, md: 6 }}
        position="sticky"
        top={0}
        zIndex={20}
      >
        <Container maxW="1400px">
          <Flex justify="space-between" align="center">
            <Flex align="center" gap={3}>
              <a
                href="https://neo4j.com"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Box
                  width={{ base: "32px", md: "40px" }}
                  height={{ base: "32px", md: "40px" }}
                  flexShrink={0}
                >
                  <svg viewBox="0 0 100 100" width="100%" height="100%">
                    <circle cx="50" cy="50" r="48" fill="#018BFF" />
                    <g fill="white">
                      <circle cx="50" cy="30" r="8" />
                      <circle cx="30" cy="65" r="8" />
                      <circle cx="70" cy="65" r="8" />
                      <line
                        x1="50"
                        y1="38"
                        x2="33"
                        y2="58"
                        stroke="white"
                        strokeWidth="3"
                      />
                      <line
                        x1="50"
                        y1="38"
                        x2="67"
                        y2="58"
                        stroke="white"
                        strokeWidth="3"
                      />
                      <line
                        x1="38"
                        y1="65"
                        x2="62"
                        y2="65"
                        stroke="white"
                        strokeWidth="3"
                      />
                    </g>
                  </svg>
                </Box>
              </a>
              <Box>
                <Heading size={{ base: "sm", md: "md" }} color="brand.600">
                  Lenny's Memory
                </Heading>
                <Text
                  color="text.muted"
                  fontSize="xs"
                  display={{ base: "none", sm: "block" }}
                >
                  Neo4j Agent Memory Demo
                </Text>
              </Box>
            </Flex>

            <Flex gap={2} align="center">
              <ToolsButton
                onClick={() => setToolsSidebarOpen(true)}
                recentCount={recentTools.length}
              />

              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSchemaDrawerOpen(true)}
                display={{ base: "none", md: "flex" }}
              >
                <LuDatabase />
                Schema
              </Button>

              <Box display={{ base: "block", md: "none" }}>
                <Menu.Root>
                  <Menu.Trigger asChild>
                    <Button variant="ghost" size="sm">
                      <LuMenu />
                    </Button>
                  </Menu.Trigger>
                  <Portal>
                    <Menu.Positioner>
                      <Menu.Content>
                        <Menu.Item
                          value="schema"
                          onClick={() => setSchemaDrawerOpen(true)}
                        >
                          <LuDatabase />
                          Schema & About
                        </Menu.Item>
                        <Menu.Item value="blog" asChild>
                          <a
                            href="https://medium.com/neo4j/meet-lennys-memory-building-context-graphs-for-ai-agents-24cb102fb91a"
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            Blog Post
                          </a>
                        </Menu.Item>
                        <Menu.Item value="github" asChild>
                          <a
                            href="https://github.com/neo4j-labs/agent-memory"
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            GitHub
                          </a>
                        </Menu.Item>
                      </Menu.Content>
                    </Menu.Positioner>
                  </Portal>
                </Menu.Root>
              </Box>

              <Box display={{ base: "none", md: "flex" }} gap={2}>
                <Button asChild variant="ghost" size="sm">
                  <a
                    href="https://medium.com/neo4j/meet-lennys-memory-building-context-graphs-for-ai-agents-24cb102fb91a"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Blog
                  </a>
                </Button>
                <Button asChild variant="ghost" size="sm">
                  <a
                    href="https://github.com/neo4j-labs/agent-memory"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    GitHub
                  </a>
                </Button>
              </Box>
            </Flex>
          </Flex>
        </Container>
      </Box>

      <Container maxW="1400px" py={0} px={0}>
        <Box
          h={{
            base: "calc(100vh - 56px)",
            md: "calc(100vh - 60px)",
          }}
        >
          <ChatInterface
            conversationHistory={conversationHistory}
            onConversationUpdate={setConversationHistory}
            onDecisionSelect={handleDecisionSelect}
            onGraphUpdate={handleGraphUpdate}
            inputValue={inputValue}
            setInputValue={setInputValue}
          />
        </Box>
      </Container>
    </Box>
  );
}
