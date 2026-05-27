import type { AttachedFile } from "./types.js";

export interface ProviderCapabilities {
  attachments?: boolean;
  systemRole?: boolean;
  toolCalls?: boolean;
}

export interface ToolFunctionDef {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

export interface ToolDef {
  type: "function";
  function: ToolFunctionDef;
}

export type ToolChoice =
  | "auto"
  | "none"
  | "required"
  | { type: "function"; function: { name: string } };

export interface ToolCall {
  id?: string;
  type: "function";
  function: { name: string; arguments: string };
  parsedArguments?: unknown;
}

export interface ProviderResult {
  provider: string;
  model: string;
  output: string;
  /** Provider-returned chain-of-thought / thinking trace, when available.
   *  e.g. DeepSeek `message.reasoning_content`, OpenAI o-series reasoning,
   *  Anthropic `<thinking>` blocks. Separated from `output` so the user-facing
   *  message can be graded for clean wording while the trace stays available
   *  to assertions about intent and tool-selection reasoning. */
  reasoningText?: string;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  error?: string;
  toolCalls?: ToolCall[];
}

/** Multi-turn chat message. Providers consume these verbatim when the caller
 *  supplies `messages`. Used by the multi-turn tool-result loop to feed model
 *  outputs and synthetic tool results back in for the next turn. */
export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  /** Required for `role: "tool"`. Matches the id of the originating tool_call. */
  tool_call_id?: string;
  /** For `role: "assistant"` messages that include tool calls from a previous turn. */
  tool_calls?: Array<{
    id?: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  /** DeepSeek-only: when replaying an assistant turn that previously emitted
   *  thinking-mode reasoning, the upstream rejects the request unless this is
   *  echoed back. Harmless on providers that ignore it. */
  reasoning_content?: string;
}

export interface CompleteChatArgs {
  system?: string;
  user: string;
  /** Full conversation. When provided, `system` and `user` are ignored. Use
   *  this for multi-turn tool-result loops. */
  messages?: ChatMessage[];
  model?: string;
  attachments?: AttachedFile[];
  tools?: ToolDef[];
  toolChoice?: ToolChoice;
  params?: Record<string, unknown>;
}

export interface Provider {
  readonly name: string;
  readonly model: string;
  readonly capabilities?: ProviderCapabilities;
  complete(prompt: string): Promise<ProviderResult>;
  completeChat?(args: CompleteChatArgs): Promise<ProviderResult>;
}

export function createStaticProvider(output: string, options: Partial<ProviderResult> = {}): Provider {
  return {
    name: options.provider ?? "static",
    model: options.model ?? "static-model",
    async complete(): Promise<ProviderResult> {
      return {
        provider: options.provider ?? "static",
        model: options.model ?? "static-model",
        output,
        latencyMs: options.latencyMs ?? 0,
        inputTokens: options.inputTokens ?? 0,
        outputTokens: options.outputTokens ?? 0,
        costUsd: options.costUsd ?? 0,
        error: options.error,
        toolCalls: options.toolCalls,
      };
    },
  };
}
