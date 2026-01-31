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
  Avatar,
} from "@chakra-ui/react";
import ReactMarkdown from "react-markdown";
import { ToolResultCard } from "./ToolResultCard";
import {
  streamChatMessage,
  getGraphData,
  type ChatMessage,
  type StreamEvent,
  type Decision,
  type GraphData,
  type AgentContext,
} from "@/lib/api";
import { LuSend, LuBot, LuUser } from "react-icons/lu";

interface ChatInterfaceProps {
  conversationHistory: ChatMessage[];
  onConversationUpdate: (messages: ChatMessage[]) => void;
  onDecisionSelect: (decision: Decision) => void;
  onGraphUpdate: (data: GraphData) => void;
  setInputValue?: (value: string) => void;
  inputValue?: string;
}

interface ToolCall {
  name: string;
  input: Record<string, unknown>;
  output?: unknown;
  graphData?: GraphData;
}

interface MessageWithGraph extends ChatMessage {
  graphData?: GraphData;
  toolCalls?: ToolCall[];
  agentContext?: AgentContext;
}

function extractEntityIds(
  toolName: string,
  input: Record<string, unknown>,
  output: unknown
): string[] {
  const ids: string[] = [];

  if (input.customer_id) ids.push(String(input.customer_id));
  if (input.account_id) ids.push(String(input.account_id));
  if (input.decision_id) ids.push(String(input.decision_id));

  if (output && typeof output === "object") {
    const result = output as Record<string, unknown>;

    if (Array.isArray(result.customers)) {
      result.customers.slice(0, 3).forEach((c: Record<string, unknown>) => {
        if (c.id) ids.push(String(c.id));
      });
    }

    if (Array.isArray(result.decisions)) {
      result.decisions.slice(0, 3).forEach((d: Record<string, unknown>) => {
        if (d.id) ids.push(String(d.id));
      });
    }

    if (Array.isArray(result.similar_decisions)) {
      result.similar_decisions
        .slice(0, 3)
        .forEach((d: Record<string, unknown>) => {
          if (d.id) ids.push(String(d.id));
        });
    }

    if (Array.isArray(result.precedents)) {
      result.precedents.slice(0, 3).forEach((p: Record<string, unknown>) => {
        if (p.id) ids.push(String(p.id));
      });
    }

    if (result.causal_chain && typeof result.causal_chain === "object") {
      const chain = result.causal_chain as Record<string, unknown>;
      if (chain.decision_id) ids.push(String(chain.decision_id));
    }
  }

  return Array.from(new Set(ids));
}

function extractGraphDataFromToolResult(output: unknown): GraphData | undefined {
  if (!output || typeof output !== "object") return undefined;
  const result = output as Record<string, unknown>;

  if (result.graph_data && typeof result.graph_data === "object") {
    const gd = result.graph_data as Record<string, unknown>;
    if (gd.nodes && gd.relationships) {
      return gd as unknown as GraphData;
    }
  }

  if (result.nodes && result.relationships) {
    return result as unknown as GraphData;
  }

  return undefined;
}

export function ChatInterface({
  conversationHistory,
  onConversationUpdate,
  onDecisionSelect,
  onGraphUpdate,
  setInputValue,
  inputValue,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<MessageWithGraph[]>([]);
  const [input, setInput] = useState(inputValue || "");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (inputValue !== undefined && inputValue !== input) {
      setInput(inputValue);
    }
  }, [inputValue]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

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
    if (setInputValue) setInputValue("");
    setIsLoading(true);
    scrollToBottom();

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
        messages.map((m) => ({ role: m.role, content: m.content }))
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
            fullContent += event.content + "\n\n";
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
            const toolIndex = toolCalls.findIndex(
              (t) => t.name === event.name && t.output === undefined
            );
            if (toolIndex !== -1) {
              const currentTool = toolCalls[toolIndex];
              toolCalls[toolIndex].output = event.output;

              const toolGraphData = extractGraphDataFromToolResult(event.output);
              if (toolGraphData) {
                toolCalls[toolIndex].graphData = toolGraphData;
                graphData = toolGraphData;
                onGraphUpdate(toolGraphData);
              } else if (event.output) {
                const entityIds = extractEntityIds(
                  currentTool.name,
                  currentTool.input,
                  event.output
                );
                if (entityIds.length > 0) {
                  getGraphData(entityIds[0], 2)
                    .then((fetchedGraphData) => {
                      if (fetchedGraphData.nodes.length > 0) {
                        toolCalls[toolIndex].graphData = fetchedGraphData;
                        graphData = fetchedGraphData;
                        onGraphUpdate(fetchedGraphData);
                        setMessages((prev) => {
                          const updated = [...prev];
                          if (updated[assistantMessageIndex]) {
                            updated[assistantMessageIndex] = {
                              ...updated[assistantMessageIndex],
                              toolCalls: [...toolCalls],
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
  }, [
    input,
    isLoading,
    messages,
    onConversationUpdate,
    onGraphUpdate,
    scrollToBottom,
    setInputValue,
  ]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Flex direction="column" h="100%">
      <Box flex={1} overflow="auto" p={{ base: 3, md: 4 }} pb={{ base: "120px", md: 4 }}>
        <VStack gap={4} align="stretch" maxW="800px" mx="auto">
          {messages.length === 0 && (
            <WelcomeMessage onSuggestionClick={(text) => setInput(text)} />
          )}

          {messages.map((message, idx) => (
            <ChatMessageBubble
              key={idx}
              message={message}
              isStreaming={
                isLoading &&
                idx === messages.length - 1 &&
                message.role === "assistant"
              }
              onDecisionClick={onDecisionSelect}
            />
          ))}

          {isLoading &&
            messages.length > 0 &&
            messages[messages.length - 1].role === "user" && (
              <Flex align="center" gap={2} px={3} py={2}>
                <Spinner size="sm" color="brand.500" />
                <Text fontSize="sm" color="text.muted">
                  Thinking...
                </Text>
              </Flex>
            )}

          <div ref={messagesEndRef} />
        </VStack>
      </Box>

      <Box
        p={{ base: 3, md: 4 }}
        borderTopWidth="1px"
        borderColor="border.default"
        bg="bg.surface"
        position={{ base: "fixed", md: "relative" }}
        bottom={{ base: 0, md: "auto" }}
        left={{ base: 0, md: "auto" }}
        right={{ base: 0, md: "auto" }}
        zIndex={10}
      >
        <Box maxW="800px" mx="auto">
          <Flex gap={2}>
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                if (setInputValue) setInputValue(e.target.value);
              }}
              onKeyDown={handleKeyDown}
              placeholder="Ask about customers, decisions, or policies..."
              rows={1}
              resize="none"
              flex={1}
              bg="bg.surface"
              borderColor="border.default"
              _focus={{
                borderColor: "brand.500",
                boxShadow: "0 0 0 1px var(--chakra-colors-brand-500)",
              }}
            />
            <IconButton
              aria-label="Send message"
              onClick={handleSend}
              colorPalette="brand"
              disabled={!input.trim() || isLoading}
            >
              <LuSend />
            </IconButton>
          </Flex>
          <Text fontSize="xs" color="text.muted" mt={2} textAlign="center">
            Press Enter to send, Shift+Enter for new line
          </Text>
        </Box>
      </Box>
    </Flex>
  );
}

function WelcomeMessage({
  onSuggestionClick,
}: {
  onSuggestionClick: (text: string) => void;
}) {
  const suggestions = [
    "Should we approve a credit limit increase for Jessica Norris? She's requesting a $25,000 limit increase.",
    "Search for customer John Walsh",
    "A customer wants to make a $15,000 wire transfer. What policies apply and are there similar past decisions?",
    "We need to override the trading limit for Katherine Miller. Find precedents for similar exceptions.",
  ];

  return (
    <Box
      bg="accent.subtle"
      p={{ base: 4, md: 6 }}
      borderRadius="xl"
      borderWidth="1px"
      borderColor="border.default"
    >
      <HStack gap={3} mb={4}>
        <Box
          p={2}
          bg="brand.500"
          borderRadius="lg"
          color="white"
        >
          <LuBot size={24} />
        </Box>
        <Box>
          <Text fontWeight="semibold" fontSize="lg">
            Welcome to Lenny's Memory
          </Text>
          <Text fontSize="sm" color="text.secondary">
            Your AI assistant for financial decision tracing
          </Text>
        </Box>
      </HStack>

      <Text fontSize="sm" color="text.secondary" mb={4}>
        I can help you search for customers, analyze decisions, find similar
        precedents, and trace causal relationships. Try one of these:
      </Text>

      <VStack align="stretch" gap={2}>
        {suggestions.map((text, idx) => (
          <Box
            key={idx}
            px={3}
            py={2}
            bg="bg.surface"
            borderRadius="lg"
            borderWidth="1px"
            borderColor="border.subtle"
            cursor="pointer"
            transition="all 0.2s"
            _hover={{
              borderColor: "brand.500",
              bg: "bg.subtle",
            }}
            onClick={() => onSuggestionClick(text)}
          >
            <Text fontSize="sm" color="text.primary">
              {text}
            </Text>
          </Box>
        ))}
      </VStack>
    </Box>
  );
}

function ChatMessageBubble({
  message,
  isStreaming,
  onDecisionClick,
}: {
  message: MessageWithGraph;
  isStreaming?: boolean;
  onDecisionClick: (decision: Decision) => void;
}) {
  const isUser = message.role === "user";

  return (
    <VStack align="stretch" gap={3}>
      <Flex
        gap={3}
        align="flex-start"
        direction={isUser ? "row-reverse" : "row"}
      >
        <Avatar.Root
          size="sm"
          bg={isUser ? "brand.500" : "secondary.500"}
          color="white"
          flexShrink={0}
        >
          <Avatar.Fallback>
            {isUser ? <LuUser size={16} /> : <LuBot size={16} />}
          </Avatar.Fallback>
        </Avatar.Root>

        <Box
          maxW={{ base: "85%", md: "75%" }}
          bg={isUser ? "brand.500" : "bg.subtle"}
          color={isUser ? "white" : "text.primary"}
          px={4}
          py={3}
          borderRadius="xl"
          borderBottomRightRadius={isUser ? "sm" : "xl"}
          borderBottomLeftRadius={isUser ? "xl" : "sm"}
        >
          {!isUser && message.agentContext && (
            <AgentContextBadge agentContext={message.agentContext} />
          )}

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
      </Flex>

      {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
        <Box pl={{ base: 0, md: "44px" }}>
          <VStack gap={3} align="stretch">
            {message.toolCalls.map((toolCall, idx) => (
              <ToolResultCard
                key={idx}
                toolName={toolCall.name}
                input={toolCall.input}
                output={toolCall.output}
                graphData={toolCall.graphData}
                isLoading={toolCall.output === undefined}
              />
            ))}
          </VStack>
        </Box>
      )}
    </VStack>
  );
}

function AgentContextBadge({ agentContext }: { agentContext: AgentContext }) {
  return (
    <Box mb={3} pb={3} borderBottom="1px solid" borderColor="border.subtle">
      <Accordion.Root collapsible defaultValue={[]}>
        <Accordion.Item value="agent-context" border="none">
          <Accordion.ItemTrigger px={0} py={2} _hover={{ bg: "transparent" }}>
            <Box flex="1" textAlign="left">
              <HStack gap={2}>
                <Badge colorPalette="teal" size="sm">
                  Agent
                </Badge>
                <Badge colorPalette="gray" size="sm" variant="subtle">
                  {agentContext.model}
                </Badge>
                <Text fontSize="xs" color="text.muted">
                  {agentContext.available_tools.length} tools available
                </Text>
              </HStack>
            </Box>
            <Accordion.ItemIndicator />
          </Accordion.ItemTrigger>
          <Accordion.ItemContent>
            <Accordion.ItemBody px={0} pb={2}>
              <VStack align="stretch" gap={3}>
                <Box p={3} bg="bg.muted" borderRadius="md">
                  <VStack align="stretch" gap={3}>
                    <Box>
                      <Text fontSize="xs" color="text.muted" fontWeight="medium" mb={1}>
                        MCP Server
                      </Text>
                      <Badge colorPalette="purple" size="sm">
                        {agentContext.mcp_server}
                      </Badge>
                    </Box>

                    <Box>
                      <Text fontSize="xs" color="text.muted" fontWeight="medium" mb={1}>
                        Available Tools
                      </Text>
                      <Flex gap={1} flexWrap="wrap">
                        {agentContext.available_tools.map((tool, idx) => (
                          <Badge key={idx} colorPalette="blue" size="sm" variant="subtle">
                            {tool}
                          </Badge>
                        ))}
                      </Flex>
                    </Box>

                    <Box>
                      <Text fontSize="xs" color="text.muted" fontWeight="medium" mb={1}>
                        System Prompt
                      </Text>
                      <Code
                        display="block"
                        whiteSpace="pre-wrap"
                        p={2}
                        borderRadius="md"
                        fontSize="xs"
                        bg="bg.surface"
                        maxH="200px"
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
