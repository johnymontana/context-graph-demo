"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Box,
  Flex,
  Text,
  VStack,
  HStack,
  Badge,
  Spinner,
  Textarea,
  IconButton,
  Code,
  Accordion,
} from "@chakra-ui/react";
import ReactMarkdown from "react-markdown";
import { InlineGraph } from "./ContextGraphView";
import {
  streamChatMessage,
  getGraphData,
  type ChatMessage,
  type StreamEvent,
  type Decision,
  type GraphData,
  type AgentContext,
} from "@/lib/api";

interface ChatInterfaceProps {
  conversationHistory: ChatMessage[];
  onConversationUpdate: (messages: ChatMessage[]) => void;
  onDecisionSelect: (decision: Decision) => void;
  onGraphUpdate: (data: GraphData) => void;
}

interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  output?: unknown;
}

interface MessageWithGraph extends ChatMessage {
  graphData?: GraphData;
  toolCalls?: ToolCall[];
  agentContext?: AgentContext;
}

// Helper function to extract entity IDs from tool results
function extractEntityIds(
  toolName: string,
  input: Record<string, unknown>,
  output: unknown,
): string[] {
  const ids: string[] = [];

  // Extract from input parameters
  if (input.customer_id) ids.push(String(input.customer_id));
  if (input.account_id) ids.push(String(input.account_id));
  if (input.decision_id) ids.push(String(input.decision_id));

  // Extract from output based on tool type
  if (output && typeof output === "object") {
    const result = output as Record<string, unknown>;

    // Handle customers array
    if (Array.isArray(result.customers)) {
      result.customers.slice(0, 3).forEach((c: Record<string, unknown>) => {
        if (c.id) ids.push(String(c.id));
      });
    }

    // Handle decisions array
    if (Array.isArray(result.decisions)) {
      result.decisions.slice(0, 3).forEach((d: Record<string, unknown>) => {
        if (d.id) ids.push(String(d.id));
      });
    }

    // Handle similar_decisions array
    if (Array.isArray(result.similar_decisions)) {
      result.similar_decisions
        .slice(0, 3)
        .forEach((d: Record<string, unknown>) => {
          if (d.id) ids.push(String(d.id));
        });
    }

    // Handle precedents array
    if (Array.isArray(result.precedents)) {
      result.precedents.slice(0, 3).forEach((p: Record<string, unknown>) => {
        if (p.id) ids.push(String(p.id));
      });
    }

    // Handle causal_chain
    if (result.causal_chain && typeof result.causal_chain === "object") {
      const chain = result.causal_chain as Record<string, unknown>;
      if (chain.decision_id) ids.push(String(chain.decision_id));
    }
  }

  return Array.from(new Set(ids)); // Remove duplicates
}

export function ChatInterface({
  conversationHistory,
  onConversationUpdate,
  onDecisionSelect,
  onGraphUpdate,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<MessageWithGraph[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [input]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: MessageWithGraph = {
      role: "user",
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Create a placeholder for the streaming assistant message
    const assistantMessageIndex = messages.length + 1;
    const assistantMessage: MessageWithGraph = {
      role: "assistant",
      content: "",
      toolCalls: [],
    };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      const toolCalls: ToolCall[] = [];
      let fullContent = "";
      let graphData: GraphData | undefined;
      let agentContext: AgentContext | undefined;

      for await (const event of streamChatMessage(
        userMessage.content,
        messages.map((m) => ({ role: m.role, content: m.content })),
      )) {
        switch (event.type) {
          case "agent_context":
            agentContext = event.context;
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                agentContext,
              };
              return updated;
            });
            break;

          case "text":
            fullContent += event.content;
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                content: fullContent,
              };
              return updated;
            });
            break;

          case "tool_use":
            toolCalls.push({ name: event.name, input: event.input });
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                toolCalls: [...toolCalls],
              };
              return updated;
            });
            break;

          case "tool_result":
            // Update the matching tool call with its output
            console.log("tool_result event:", event);
            console.log("Looking for tool with name:", event.name);
            console.log(
              "Current toolCalls:",
              JSON.stringify(toolCalls, null, 2),
            );
            const toolIndex = toolCalls.findIndex(
              (t) => t.name === event.name && t.output === undefined,
            );
            console.log("Found toolIndex:", toolIndex);
            if (toolIndex !== -1) {
              const currentTool = toolCalls[toolIndex];
              toolCalls[toolIndex].output = event.output;
              console.log("Updated toolCall output:", toolCalls[toolIndex]);

              // Check if this tool returned graph data directly
              let foundGraphData = false;
              if (event.output && typeof event.output === "object") {
                const result = event.output as Record<string, unknown>;
                // Check for graph_data field (new format from agent tools)
                if (
                  result.graph_data &&
                  typeof result.graph_data === "object"
                ) {
                  const gd = result.graph_data as Record<string, unknown>;
                  if (gd.nodes && gd.relationships) {
                    graphData = gd as unknown as GraphData;
                    onGraphUpdate(graphData);
                    foundGraphData = true;
                  }
                }
                // Also check for direct nodes/relationships (legacy format)
                else if (result.nodes && result.relationships) {
                  graphData = result as unknown as GraphData;
                  onGraphUpdate(graphData);
                  foundGraphData = true;
                }
              }

              // If no graph data was returned, try to fetch it based on entity IDs
              if (!foundGraphData && event.output) {
                const entityIds = extractEntityIds(
                  currentTool.name,
                  currentTool.input,
                  event.output,
                );
                if (entityIds.length > 0) {
                  // Fetch graph data for the first entity ID found
                  getGraphData(entityIds[0], 2)
                    .then((fetchedGraphData) => {
                      if (fetchedGraphData.nodes.length > 0) {
                        graphData = fetchedGraphData;
                        onGraphUpdate(fetchedGraphData);
                        // Update message with graph data
                        setMessages((prev) => {
                          const updated = [...prev];
                          if (updated[assistantMessageIndex]) {
                            updated[assistantMessageIndex] = {
                              ...updated[assistantMessageIndex],
                              graphData: fetchedGraphData,
                            };
                          }
                          return updated;
                        });
                      }
                    })
                    .catch((err) => {
                      console.error("Failed to fetch graph data:", err);
                    });
                }
              }

              setMessages((prev) => {
                const updated = [...prev];
                updated[assistantMessageIndex] = {
                  ...updated[assistantMessageIndex],
                  toolCalls: [...toolCalls],
                  graphData,
                };
                return updated;
              });
            }
            break;

          case "done":
            // Final update with complete data
            // Use local toolCalls array which has outputs populated from tool_result events
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                content: fullContent,
                toolCalls: toolCalls,
                graphData,
                agentContext,
              };
              return updated;
            });
            break;

          case "error":
            setMessages((prev) => {
              const updated = [...prev];
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                content: `Error: ${event.error}`,
              };
              return updated;
            });
            break;
        }
      }

      // Update conversation history
      const finalMessage: MessageWithGraph = {
        role: "assistant",
        content: fullContent,
        toolCalls,
        graphData,
        agentContext,
      };
      onConversationUpdate([...messages, userMessage, finalMessage]);
    } catch (error) {
      console.error("Failed to send message:", error);
      setMessages((prev) => {
        const updated = [...prev];
        updated[assistantMessageIndex] = {
          role: "assistant",
          content:
            "Sorry, I encountered an error processing your request. Please try again.",
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, messages, onConversationUpdate, onGraphUpdate]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Flex direction="column" h="100%">
      {/* Messages Area */}
      <Box flex={1} overflow="auto" p={4}>
        <VStack gap={4} align="stretch">
          {/* Welcome message */}
          {messages.length === 0 && (
            <Box
              bg="brand.50"
              _dark={{ bg: "brand.900", borderColor: "brand.700" }}
              p={4}
              borderRadius="lg"
              borderWidth="1px"
              borderColor="brand.200"
            >
              <Text fontWeight="medium" mb={2}>
                Welcome to Context Graph Demo
              </Text>
              <Text
                fontSize="sm"
                color="gray.600"
                _dark={{ color: "gray.400" }}
              >
                I can help you search for customers, analyze decisions, find
                similar precedents, and trace causal relationships. Try asking:
              </Text>
              <VStack align="start" mt={3} gap={1}>
                <SuggestionChip
                  text="Should we approve a credit limit increase for Jessica Norris? She's requesting a $25,000 limit increase."
                  onClick={() =>
                    setInput(
                      "Should we approve a credit limit increase for Jessica Norris? She's requesting a $25,000 limit increase.",
                    )
                  }
                />
                <SuggestionChip
                  text="Search for customer John Walsh"
                  onClick={() => setInput("Search for customer John Walsh")}
                />
                <SuggestionChip
                  text="A customer wants to make a $15,000 wire transfer. What policies apply and are there similar past decisions?"
                  onClick={() =>
                    setInput(
                      "A customer wants to make a $15,000 wire transfer. What policies apply and are there similar past decisions?",
                    )
                  }
                />
                <SuggestionChip
                  text="We need to override the trading limit for Katherine Miller. Find precedents for similar exceptions."
                  onClick={() =>
                    setInput(
                      "We need to override the trading limit for Katherine Miller. Find precedents for similar exceptions.",
                    )
                  }
                />
              </VStack>
            </Box>
          )}

          {/* Chat messages */}
          {messages.map((message, idx) => (
            <Box key={idx} mb={message.role === "assistant" ? 4 : 0}>
              <ChatMessageBubble
                message={message}
                isStreaming={
                  isLoading &&
                  idx === messages.length - 1 &&
                  message.role === "assistant"
                }
                onDecisionClick={onDecisionSelect}
                onNodeClick={(nodeId, labels) => {
                  console.log("Node clicked in chat:", nodeId, labels);
                }}
              />
            </Box>
          ))}

          {/* Loading indicator - only show when waiting for first response */}
          {isLoading &&
            messages.length > 0 &&
            messages[messages.length - 1].role === "user" && (
              <Flex align="center" gap={2} p={3}>
                <Spinner size="sm" color="brand.500" />
                <Text fontSize="sm" color="gray.500">
                  Thinking...
                </Text>
              </Flex>
            )}

          <div ref={messagesEndRef} />
        </VStack>
      </Box>

      {/* Input Area */}
      <Box p={4} borderTopWidth="1px" borderColor="border.default">
        <Flex gap={2}>
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about customers, decisions, or policies..."
            rows={1}
            resize="none"
            flex={1}
          />
          <IconButton
            aria-label="Send message"
            onClick={handleSend}
            colorPalette="brand"
          >
            <SendIcon />
          </IconButton>
        </Flex>
      </Box>
    </Flex>
  );
}

// Helper to format JSON for display
function formatJSON(obj: unknown): string {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

// Tool call display component
function ToolCallDisplay({
  toolCall,
  index,
}: {
  toolCall: ToolCall;
  index: number;
}) {
  return (
    <Box
      key={index}
      mt={2}
      borderLeft="3px solid"
      borderColor="blue.400"
      pl={3}
    >
      <HStack mb={2}>
        <Badge colorPalette="blue" fontSize="xs">
          Tool Call
        </Badge>
        <Text fontWeight="bold" fontSize="sm">
          {toolCall.name.replace("mcp__graph__", "")}
        </Text>
      </HStack>

      <Box mb={2}>
        <Text fontSize="xs" color="gray.600" fontWeight="semibold" mb={1}>
          Arguments:
        </Text>
        <Code
          display="block"
          whiteSpace="pre-wrap"
          p={2}
          borderRadius="md"
          fontSize="xs"
          bg="gray.50"
          maxH="200px"
          overflowY="auto"
        >
          {formatJSON(toolCall.input)}
        </Code>
      </Box>

      {toolCall.output !== undefined && (
        <Box>
          <Text fontSize="xs" color="gray.600" fontWeight="semibold" mb={1}>
            Output:
          </Text>
          <Code
            display="block"
            whiteSpace="pre-wrap"
            p={2}
            borderRadius="md"
            fontSize="xs"
            bg="gray.50"
            maxH="300px"
            overflowY="auto"
          >
            {formatJSON(toolCall.output)}
          </Code>
        </Box>
      )}
    </Box>
  );
}

// Agent context disclosure component
function AgentContextDisclosure({
  agentContext,
}: {
  agentContext: AgentContext;
}) {
  return (
    <Box mb={3} pb={3} borderBottom="1px solid" borderColor="gray.200">
      <Accordion.Root collapsible defaultValue={[]}>
        <Accordion.Item value="agent-context" border="none">
          <Accordion.ItemTrigger px={0} py={2} _hover={{ bg: "transparent" }}>
            <Box flex="1" textAlign="left">
              <HStack>
                <Badge colorPalette="teal" fontSize="xs">
                  Agent Context
                </Badge>
                <Text fontSize="xs" color="gray.600">
                  View system prompt and configuration
                </Text>
              </HStack>
            </Box>
            <Accordion.ItemIndicator />
          </Accordion.ItemTrigger>
          <Accordion.ItemContent>
            <Accordion.ItemBody px={0} pb={2}>
              <VStack align="stretch" gap={3}>
                <Box
                  p={3}
                  bg="teal.50"
                  borderRadius="md"
                  border="1px solid"
                  borderColor="teal.200"
                >
                  <VStack align="stretch" gap={3}>
                    {/* Model */}
                    <Box>
                      <Text
                        fontSize="xs"
                        color="gray.600"
                        fontWeight="semibold"
                        mb={1}
                      >
                        Model:
                      </Text>
                      <Badge colorPalette="teal" fontSize="xs">
                        {agentContext.model}
                      </Badge>
                    </Box>

                    {/* MCP Server */}
                    <Box>
                      <Text
                        fontSize="xs"
                        color="gray.600"
                        fontWeight="semibold"
                        mb={1}
                      >
                        MCP Server:
                      </Text>
                      <Badge colorPalette="purple" fontSize="xs">
                        {agentContext.mcp_server}
                      </Badge>
                    </Box>

                    {/* Available Tools */}
                    <Box>
                      <Text
                        fontSize="xs"
                        color="gray.600"
                        fontWeight="semibold"
                        mb={1}
                      >
                        Available Tools:
                      </Text>
                      <Flex gap={1} flexWrap="wrap">
                        {agentContext.available_tools.map((tool, idx) => (
                          <Badge key={idx} colorPalette="blue" fontSize="xs">
                            {tool}
                          </Badge>
                        ))}
                      </Flex>
                    </Box>

                    {/* System Prompt */}
                    <Box>
                      <Text
                        fontSize="xs"
                        color="gray.600"
                        fontWeight="semibold"
                        mb={1}
                      >
                        System Prompt:
                      </Text>
                      <Code
                        display="block"
                        whiteSpace="pre-wrap"
                        p={2}
                        borderRadius="md"
                        fontSize="xs"
                        bg="white"
                        maxH="300px"
                        overflowY="auto"
                      >
                        {agentContext.system_prompt}
                      </Code>
                    </Box>
                  </VStack>
                </Box>
              </VStack>
            </Accordion.ItemBody>
          </Accordion.ItemContent>
        </Accordion.Item>
      </Accordion.Root>
    </Box>
  );
}

// Tool calls disclosure component
function ToolCallsDisclosure({ toolCalls }: { toolCalls: ToolCall[] }) {
  if (!toolCalls || toolCalls.length === 0) return null;

  return (
    <Box mb={3} pb={3} borderBottom="1px solid" borderColor="gray.200">
      <Accordion.Root collapsible defaultValue={[]}>
        <Accordion.Item value="tool-calls" border="none">
          <Accordion.ItemTrigger px={0} py={2} _hover={{ bg: "transparent" }}>
            <Box flex="1" textAlign="left">
              <HStack>
                <Badge colorPalette="purple" fontSize="xs">
                  Tool Calls ({toolCalls.length})
                </Badge>
                <Text fontSize="xs" color="gray.600">
                  View tool parameters and results
                </Text>
              </HStack>
            </Box>
            <Accordion.ItemIndicator />
          </Accordion.ItemTrigger>
          <Accordion.ItemContent>
            <Accordion.ItemBody px={0} pb={2}>
              <VStack align="stretch" gap={3}>
                {toolCalls.map((toolCall, index) => (
                  <Box
                    key={index}
                    p={3}
                    bg="purple.50"
                    borderRadius="md"
                    border="1px solid"
                    borderColor="purple.200"
                  >
                    <ToolCallDisplay toolCall={toolCall} index={index} />
                  </Box>
                ))}
              </VStack>
            </Accordion.ItemBody>
          </Accordion.ItemContent>
        </Accordion.Item>
      </Accordion.Root>
    </Box>
  );
}

// Chat message bubble component
function ChatMessageBubble({
  message,
  isStreaming,
  onDecisionClick,
  onNodeClick,
}: {
  message: MessageWithGraph;
  isStreaming?: boolean;
  onDecisionClick: (decision: Decision) => void;
  onNodeClick: (nodeId: string, labels: string[]) => void;
}) {
  const isUser = message.role === "user";

  return (
    <Box
      alignSelf={isUser ? "flex-end" : "flex-start"}
      maxW="85%"
      w={
        message.graphData ||
        (!isUser && (message.agentContext || message.toolCalls?.length))
          ? "100%"
          : "auto"
      }
      minW={
        !isUser && (message.agentContext || message.toolCalls?.length)
          ? "60%"
          : "auto"
      }
    >
      <Box
        bg={isUser ? "brand.500" : "bg.subtle"}
        color={isUser ? "white" : "inherit"}
        px={4}
        py={3}
        borderRadius="lg"
        borderBottomRightRadius={isUser ? "sm" : "lg"}
        borderBottomLeftRadius={isUser ? "lg" : "sm"}
      >
        {/* Agent context disclosure (for assistant messages) */}
        {!isUser && message.agentContext && (
          <AgentContextDisclosure agentContext={message.agentContext} />
        )}

        {/* Tool calls disclosure (for assistant messages) */}
        {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
          <ToolCallsDisclosure toolCalls={message.toolCalls} />
        )}

        {/* Message content */}
        <Flex align="flex-start" gap={2}>
          {isUser ? (
            <Text whiteSpace="pre-wrap" fontSize="sm" flex={1}>
              {message.content}
            </Text>
          ) : (
            <Box flex={1} fontSize="sm" className="markdown-content">
              <ReactMarkdown
                components={{
                  p: ({ children }) => (
                    <Text mb={2} _last={{ mb: 0 }}>
                      {children}
                    </Text>
                  ),
                  strong: ({ children }) => (
                    <Text as="strong" fontWeight="bold">
                      {children}
                    </Text>
                  ),
                  em: ({ children }) => (
                    <Text as="em" fontStyle="italic">
                      {children}
                    </Text>
                  ),
                  ul: ({ children }) => (
                    <Box as="ul" pl={4} mb={2}>
                      {children}
                    </Box>
                  ),
                  ol: ({ children }) => (
                    <Box as="ol" pl={4} mb={2}>
                      {children}
                    </Box>
                  ),
                  li: ({ children }) => (
                    <Box as="li" mb={1}>
                      {children}
                    </Box>
                  ),
                  code: ({ children, className }) => {
                    const isInline = !className;
                    return isInline ? (
                      <Code fontSize="xs" px={1}>
                        {children}
                      </Code>
                    ) : (
                      <Box
                        as="pre"
                        bg="bg.emphasized"
                        p={3}
                        borderRadius="md"
                        overflow="auto"
                        mb={2}
                        fontSize="xs"
                      >
                        <code>{children}</code>
                      </Box>
                    );
                  },
                  h1: ({ children }) => (
                    <Text fontSize="lg" fontWeight="bold" mb={2} mt={3}>
                      {children}
                    </Text>
                  ),
                  h2: ({ children }) => (
                    <Text fontSize="md" fontWeight="bold" mb={2} mt={3}>
                      {children}
                    </Text>
                  ),
                  h3: ({ children }) => (
                    <Text fontSize="sm" fontWeight="bold" mb={1} mt={2}>
                      {children}
                    </Text>
                  ),
                  blockquote: ({ children }) => (
                    <Box
                      borderLeftWidth={3}
                      borderLeftColor="gray.300"
                      pl={3}
                      my={2}
                      color="gray.600"
                    >
                      {children}
                    </Box>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </Box>
          )}
          {isStreaming && <Spinner size="xs" color="brand.500" />}
        </Flex>
      </Box>

      {/* Inline graph visualization */}
      {message.graphData && message.graphData.nodes.length > 0 && (
        <Box mt={2}>
          <InlineGraph
            graphData={message.graphData}
            height="250px"
            onNodeClick={onNodeClick}
          />
          <Text fontSize="xs" color="gray.500" mt={1}>
            {message.graphData.nodes.length} nodes,{" "}
            {message.graphData.relationships.length} relationships
          </Text>
        </Box>
      )}
    </Box>
  );
}

// Suggestion chip component
function SuggestionChip({
  text,
  onClick,
}: {
  text: string;
  onClick: () => void;
}) {
  return (
    <Text
      fontSize="sm"
      color="brand.600"
      cursor="pointer"
      _hover={{ textDecoration: "underline" }}
      onClick={onClick}
    >
      → {text}
    </Text>
  );
}

// Send icon component
function SendIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}
