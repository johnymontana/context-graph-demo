"use client";

import {
  Box,
  Flex,
  Text,
  Badge,
  HStack,
  Popover,
  Portal,
  IconButton,
} from "@chakra-ui/react";
import { LuInfo, LuCircle, LuArrowRight } from "react-icons/lu";

interface DataModelBadgeProps {
  nodeTypes: string[];
  relationshipTypes: string[];
  description?: string;
}

const NODE_COLORS: Record<string, string> = {
  Person: "blue",
  Account: "green",
  Transaction: "orange",
  Decision: "purple",
  Organization: "red",
  Policy: "cyan",
  Employee: "indigo",
  Alert: "yellow",
  Community: "orange",
  SupportTicket: "blue",
  Exception: "yellow",
  Escalation: "purple",
  Precedent: "pink",
};

export function DataModelBadge({
  nodeTypes,
  relationshipTypes,
  description,
}: DataModelBadgeProps) {
  if (nodeTypes.length === 0 && relationshipTypes.length === 0) {
    return null;
  }

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <Flex
          align="center"
          gap={1}
          px={2}
          py={1}
          bg="bg.subtle"
          borderRadius="full"
          cursor="pointer"
          _hover={{ bg: "bg.emphasized" }}
          transition="background 0.2s"
          fontSize="xs"
          color="text.secondary"
        >
          <LuInfo size={12} />
          <Text>
            {nodeTypes.length} types &bull; {relationshipTypes.length} rels
          </Text>
        </Flex>
      </Popover.Trigger>
      <Portal>
        <Popover.Positioner>
          <Popover.Content maxW="300px">
            <Popover.Arrow>
              <Popover.ArrowTip />
            </Popover.Arrow>
            <Popover.Header fontWeight="medium" fontSize="sm">
              Data Model Context
            </Popover.Header>
            <Popover.Body>
              <Box mb={3}>
                <Text fontSize="xs" color="text.muted" mb={2}>
                  Node Types
                </Text>
                <Flex gap={1} flexWrap="wrap">
                  {nodeTypes.map((type) => (
                    <Badge
                      key={type}
                      size="sm"
                      colorPalette={NODE_COLORS[type] || "gray"}
                    >
                      {type}
                    </Badge>
                  ))}
                </Flex>
              </Box>
              {relationshipTypes.length > 0 && (
                <Box mb={3}>
                  <Text fontSize="xs" color="text.muted" mb={2}>
                    Relationships
                  </Text>
                  <Flex gap={1} flexWrap="wrap">
                    {relationshipTypes.map((type) => (
                      <Badge key={type} size="sm" variant="outline">
                        {type}
                      </Badge>
                    ))}
                  </Flex>
                </Box>
              )}
              {description && (
                <Text fontSize="xs" color="text.secondary">
                  {description}
                </Text>
              )}
            </Popover.Body>
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
}

export function InlineDataModel({
  nodeTypes,
  relationshipTypes,
}: {
  nodeTypes: string[];
  relationshipTypes: string[];
}) {
  return (
    <Box
      bg="bg.subtle"
      borderRadius="lg"
      p={3}
      borderWidth="1px"
      borderColor="border.subtle"
    >
      <Text fontSize="xs" fontWeight="medium" color="text.muted" mb={2}>
        Query Traversal
      </Text>
      <HStack gap={2} flexWrap="wrap">
        {nodeTypes.map((type, i) => (
          <HStack key={type} gap={1}>
            <Badge size="sm" colorPalette={NODE_COLORS[type] || "gray"}>
              {type}
            </Badge>
            {i < nodeTypes.length - 1 && relationshipTypes[i] && (
              <HStack gap={1} color="text.muted">
                <Box
                  w={4}
                  h="1px"
                  bg="currentColor"
                  display={{ base: "none", sm: "block" }}
                />
                <Text fontSize="xs" fontFamily="mono">
                  {relationshipTypes[i]}
                </Text>
                <LuArrowRight size={10} />
              </HStack>
            )}
          </HStack>
        ))}
      </HStack>
    </Box>
  );
}
