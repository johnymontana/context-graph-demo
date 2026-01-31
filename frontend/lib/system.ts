import { createSystem, defaultConfig, defineConfig } from "@chakra-ui/react";

const config = defineConfig({
  theme: {
    tokens: {
      fonts: {
        heading: { value: '"Public Sans", system-ui, sans-serif' },
        body: { value: '"Public Sans", system-ui, sans-serif' },
      },
      colors: {
        brand: {
          50: { value: "#E8F4FF" },
          100: { value: "#C5E4FF" },
          200: { value: "#9DD0FF" },
          300: { value: "#6BB8FF" },
          400: { value: "#3D9EFF" },
          500: { value: "#018BFF" },
          600: { value: "#0070CC" },
          700: { value: "#005599" },
          800: { value: "#003B66" },
          900: { value: "#002033" },
        },
        neo4j: {
          blue: { value: "#018BFF" },
          green: { value: "#10B860" },
          darkBlue: { value: "#002033" },
        },
        secondary: {
          50: { value: "#E8FAF0" },
          100: { value: "#C5F2D9" },
          200: { value: "#9DE9BF" },
          300: { value: "#6BD9A0" },
          400: { value: "#3DC97F" },
          500: { value: "#10B860" },
          600: { value: "#0D944D" },
          700: { value: "#0A703A" },
          800: { value: "#064C27" },
          900: { value: "#032814" },
        },
        decision: {
          credit: { value: "#10B860" },
          fraud: { value: "#EF4444" },
          trading: { value: "#3B82F6" },
          exception: { value: "#F59E0B" },
          escalation: { value: "#8B5CF6" },
        },
        node: {
          person: { value: "#3B82F6" },
          account: { value: "#10B860" },
          transaction: { value: "#F59E0B" },
          decision: { value: "#8B5CF6" },
          organization: { value: "#EF4444" },
          policy: { value: "#06B6D4" },
          memory: { value: "#EC4899" },
          community: { value: "#F97316" },
          employee: { value: "#6366F1" },
          alert: { value: "#FBBF24" },
        },
      },
    },
    semanticTokens: {
      colors: {
        "bg.canvas": {
          value: { _light: "{colors.gray.50}", _dark: "{colors.gray.950}" },
        },
        "bg.surface": {
          value: { _light: "white", _dark: "{colors.gray.900}" },
        },
        "bg.subtle": {
          value: { _light: "{colors.gray.100}", _dark: "{colors.gray.800}" },
        },
        "bg.emphasized": {
          value: { _light: "{colors.gray.200}", _dark: "{colors.gray.700}" },
        },
        "bg.muted": {
          value: { _light: "{colors.gray.50}", _dark: "{colors.gray.900}" },
        },
        "border.default": {
          value: { _light: "{colors.gray.200}", _dark: "{colors.gray.700}" },
        },
        "border.subtle": {
          value: { _light: "{colors.gray.100}", _dark: "{colors.gray.800}" },
        },
        "text.primary": {
          value: { _light: "{colors.gray.900}", _dark: "{colors.gray.50}" },
        },
        "text.secondary": {
          value: { _light: "{colors.gray.600}", _dark: "{colors.gray.400}" },
        },
        "text.muted": {
          value: { _light: "{colors.gray.500}", _dark: "{colors.gray.500}" },
        },
        "accent.default": {
          value: { _light: "{colors.brand.500}", _dark: "{colors.brand.400}" },
        },
        "accent.subtle": {
          value: { _light: "{colors.brand.50}", _dark: "{colors.brand.900}" },
        },
      },
    },
  },
});

export const system = createSystem(defaultConfig, config);
