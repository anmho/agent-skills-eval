import path from "node:path";
import type { Provider } from "./provider.js";
import type { ProviderResult } from "./provider.js";
import { writeRunArtifacts } from "./artifacts.js";
import { gradeOutputs } from "./grade.js";
import type {
  AgentSkillsEval,
  AttachedFile,
  GradingJson,
  Skill,
  SkillsEvent,
  ToolCall,
  ToolChoice,
  ToolDef,
} from "./types.js";
import { attachedFileXml, readAttachedFile, slugify } from "./fs-utils.js";

export type RunMode = "with_skill" | "without_skill";

export interface RunEvalArgs {
  skill: Skill;
  eval: AgentSkillsEval;
  modes: RunMode[];
  target: { model: string; provider: Provider };
  judge: { model: string; provider: Provider };
  workspace: string;
  iteration: number;
  gradingPrompt?: string;
  index?: number;
  evalRootDir?: string;
  /**
   * Caller-level inference param defaults for the target model. Lowest
   * precedence: skill `defaults.target.params` and eval `params` override.
   */
  targetParams?: Record<string, unknown>;
  /** Caller-level defaults for the judge model. */
  judgeParams?: Record<string, unknown>;
  /** Maximum number of model turns in the tool-result loop. Each turn = one
   *  call to the model. Default 6. After this, the loop stops even if the
   *  model is still asking for more tool calls and the result is marked
   *  truncated. Set to 1 to recover the legacy single-shot behaviour. */
  maxToolTurns?: number;
  /** Receives eval-start / eval-end events as each mode runs. */
  onEvent?: (event: SkillsEvent) => void;
}

export interface RunEvalResult {
  slug: string;
  modes: Record<RunMode, {
    outputDir: string;
    timing: { total_tokens: number; duration_ms: number };
    grading: GradingJson;
    rawOutput: string;
    toolCalls?: ToolCall[];
    /** System message sent to the target model (only set in `with_skill`). */
    system?: string;
    /** User message sent to the target model. */
    user: string;
    /** Number of attached `evals[].files`. */
    fileCount: number;
    /** Final prompt sent to the judge for grading. */
    judgePrompt: string;
    /** Tools made available for this run, if any. */
    tools?: ToolDef[];
    toolChoice?: ToolChoice;
  }>;
}

function evalSlug(evalCase: AgentSkillsEval, index = 0): string {
  const source = evalCase.name ?? (evalCase.id !== undefined ? `eval-${String(evalCase.id)}` : `eval-${index + 1}`);
  const slug = slugify(source, `eval-${index + 1}`);
  return slug.startsWith("eval-") ? slug : `eval-${slug}`;
}

function renderSkillSystemMessage(skill: Skill): string {
  const parts = [
    `<skill name="${skill.name}">`,
    `<description>${skill.description ?? ""}</description>`,
    `<instructions>`,
    skill.skillMd,
    `</instructions>`,
  ];

  if (skill.references.length > 0) {
    parts.push(`<references>`);
    for (const ref of skill.references) parts.push(attachedFileXml("reference", ref));
    parts.push(`</references>`);
  }

  if (skill.scripts.length > 0) {
    parts.push(`<scripts>`);
    for (const script of skill.scripts) parts.push(attachedFileXml("script", script));
    parts.push(`</scripts>`);
  }

  parts.push(`</skill>`);
  return parts.join("\n");
}

function readEvalFiles(skill: Skill, evalCase: AgentSkillsEval): AttachedFile[] {
  return (evalCase.files ?? []).map((relativePath) =>
    readAttachedFile(skill.dir, relativePath)
  );
}

function inlineFiles(user: string, files: AttachedFile[]): string {
  if (files.length === 0) return user;
  return [
    ...files.map((file) => attachedFileXml("file", file)),
    "---USER PROMPT---",
    user,
  ].join("\n\n");
}

async function completeWithFallback(args: {
  provider: Provider;
  system?: string;
  user: string;
  attachments: AttachedFile[];
  tools?: ToolDef[];
  toolChoice?: ToolChoice;
  params?: Record<string, unknown>;
}): Promise<ProviderResult> {
  const { provider, system, tools, toolChoice, params } = args;
  let user = args.user;
  let attachments: AttachedFile[] | undefined;

  if (provider.capabilities?.attachments) {
    attachments = args.attachments;
  } else {
    user = inlineFiles(user, args.attachments);
  }

  if (provider.completeChat && provider.capabilities?.systemRole) {
    return provider.completeChat({
      system,
      user,
      attachments,
      tools,
      toolChoice,
      params,
    });
  }

  const merged = [system, "", "---USER REQUEST---", user].filter(Boolean).join("\n");
  return provider.complete(merged);
}

/** Synthesize a plausible tool result so the model can produce its final
 *  user-facing turn. Eval authors can opt into richer mocks per-tool via
 *  `defaults.tool_mocks[name]`, but the default is just a success stub so the
 *  conversation can proceed without external side effects. */
function defaultToolResult(call: ToolCall): string {
  return JSON.stringify({ ok: true, tool: call.function.name });
}

/** Resolve a per-tool mock from `defaults.tool_mocks` to the JSON string the
 *  loop will hand back as the tool's result. Supports:
 *    - `"echo"` — echo back the parsed arguments (useful for "what did the
 *      model just say" assertions).
 *    - `"echo_gmail_url"` — build a mail.google.com deeplink from `thread_id`
 *      in the parsed arguments (Gmail thread URL eval fixtures).
 *    - any JSON-serializable object — stringified verbatim.
 *  Anything else falls back to the default ok stub. */
function resolveToolMock(mock: unknown, call: ToolCall): string {
  if (mock === "echo") {
    return JSON.stringify(call.parsedArguments ?? {});
  }
  if (mock === "echo_gmail_url") {
    const args = (call.parsedArguments ?? {}) as Record<string, unknown>;
    const threadId =
      typeof args.thread_id === "string"
        ? args.thread_id
        : typeof args.threadId === "string"
          ? args.threadId
          : "";
    const accountIndex =
      typeof args.account_index === "number" ? args.account_index : 0;
    const url = threadId
      ? `https://mail.google.com/mail/u/${accountIndex}/#all/${threadId}`
      : "";
    return JSON.stringify({ thread_id: threadId, url });
  }
  if (mock && typeof mock === "object") {
    return JSON.stringify(mock);
  }
  return defaultToolResult(call);
}

interface MultiTurnResult {
  /** Final assistant `content` string (the user-facing reply, after all tool
   *  calls were resolved). Empty string when the model never produced text. */
  output: string;
  /** All `tool_calls` emitted across every turn, in chronological order. */
  toolCalls: ToolCall[];
  /** Reasoning trace from the FINAL turn — the trace the user-facing reply was
   *  produced from. Per-turn traces are not retained today. */
  reasoningText?: string;
  /** Aggregated timing/usage stats summed across turns. */
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  provider: string;
  model: string;
  error?: string;
  /** Number of model turns actually taken (initial + tool-result turns). */
  turns: number;
  /** True when the loop hit `maxTurns` while the model was still asking for
   *  more tool calls — final `output` will likely be empty in that case. */
  truncated: boolean;
}

/** Drive the chat through tool-result rounds until the model produces a
 *  user-facing message (or until `maxTurns` is exhausted). Each round:
 *    1. Send the current message stack to the provider.
 *    2. If the response has `tool_calls`, append the assistant message and one
 *       `role:"tool"` message per call (with a synthesized result), then loop.
 *    3. Otherwise the model produced its final user-facing reply — return.
 *
 *  `tool_mocks` lets an eval author override the synthetic result for specific
 *  tools (e.g. to return a Gmail thread list so the model can pick one). The
 *  default `{ ok: true }` stub is fine for most "did the model call the right
 *  tool?" cases. */
async function completeWithToolLoop(args: {
  provider: Provider;
  system?: string;
  user: string;
  attachments: AttachedFile[];
  tools?: ToolDef[];
  toolChoice?: ToolChoice;
  params?: Record<string, unknown>;
  maxTurns: number;
  toolMocks?: Record<string, unknown>;
}): Promise<MultiTurnResult> {
  const provider = args.provider;
  const supportsMessages =
    Boolean(provider.completeChat) && Boolean(provider.capabilities?.systemRole);

  // If the provider doesn't support a real chat shape, fall through to the
  // single-shot path. Tool loops would be meaningless there.
  if (!supportsMessages) {
    const single = await completeWithFallback({
      provider,
      system: args.system,
      user: args.user,
      attachments: args.attachments,
      tools: args.tools,
      toolChoice: args.toolChoice,
      params: args.params,
    });
    return {
      output: single.output,
      toolCalls: single.toolCalls ?? [],
      reasoningText: single.reasoningText,
      latencyMs: single.latencyMs,
      inputTokens: single.inputTokens,
      outputTokens: single.outputTokens,
      costUsd: single.costUsd,
      provider: single.provider,
      model: single.model,
      error: single.error,
      turns: 1,
      truncated: false,
    };
  }

  // Inline attachments into the first user turn if the provider can't take
  // attachments natively — matches the legacy single-shot behaviour.
  let user = args.user;
  let attachments: AttachedFile[] | undefined;
  if (provider.capabilities?.attachments) {
    attachments = args.attachments;
  } else {
    user = inlineFiles(user, args.attachments);
  }

  const messages: Array<{
    role: "system" | "user" | "assistant" | "tool";
    content: string | null;
    tool_call_id?: string;
    tool_calls?: Array<{
      id?: string;
      type: "function";
      function: { name: string; arguments: string };
    }>;
    reasoning_content?: string;
  }> = [];
  if (args.system) messages.push({ role: "system", content: args.system });
  messages.push({ role: "user", content: user });

  const aggregateCalls: ToolCall[] = [];
  // Collect the reasoning trace from every turn so the judge can see the
  // model's intent at the moment it picked a tool, not just whatever it
  // thought after the synthetic tool result came back.
  const reasoningTurns: string[] = [];
  let lastResult: ProviderResult | undefined;
  let totalLatency = 0;
  let totalInput = 0;
  let totalOutput = 0;
  let totalCost = 0;
  let turns = 0;
  let truncated = false;

  for (let turn = 0; turn < args.maxTurns; turn++) {
    turns++;
    const result = await provider.completeChat!({
      user: "",
      messages,
      attachments,
      tools: args.tools,
      toolChoice: args.toolChoice,
      params: args.params,
    });
    lastResult = result;
    totalLatency += result.latencyMs;
    totalInput += result.inputTokens;
    totalOutput += result.outputTokens;
    totalCost += result.costUsd;

    if (result.error) break;

    if (result.reasoningText && result.reasoningText.trim().length > 0) {
      reasoningTurns.push(result.reasoningText);
    }

    const calls = result.toolCalls ?? [];
    aggregateCalls.push(...calls);

    if (calls.length === 0) {
      // Model produced a user-facing reply (or empty). Loop done.
      break;
    }

    // Append the assistant's tool-call turn, then synthesize per-call results.
    // OpenAI's protocol requires the assistant message to include the same
    // `tool_calls` array, and each subsequent `role:"tool"` message to carry
    // the matching `tool_call_id`. Without those ids the upstream rejects the
    // request, so when an id is missing we fall back to a synthetic one.
    const normalizedCalls = calls.map((c, i) => ({
      id: c.id ?? `synthetic-${turn}-${i}`,
      type: "function" as const,
      function: { name: c.function.name, arguments: c.function.arguments },
    }));
    messages.push({
      role: "assistant",
      content: result.output ?? "",
      tool_calls: normalizedCalls,
      // DeepSeek (and other thinking-mode providers) require the prior
      // reasoning trace to be echoed back when we replay the assistant turn,
      // otherwise the next request 400s. Sending it on providers that don't
      // recognise the field is harmless — they ignore unknown keys.
      ...(result.reasoningText ? { reasoning_content: result.reasoningText } : {}),
    });
    for (let i = 0; i < normalizedCalls.length; i++) {
      const c = normalizedCalls[i];
      const originalCall = calls[i];
      const mock = args.toolMocks?.[c.function.name];
      const content = mock !== undefined
        ? resolveToolMock(mock, originalCall)
        : defaultToolResult(originalCall);
      messages.push({
        role: "tool",
        tool_call_id: c.id,
        content,
      });
    }

    if (turn === args.maxTurns - 1) {
      truncated = true;
    }
  }

  const concatenatedReasoning = reasoningTurns.length === 0
    ? undefined
    : reasoningTurns.length === 1
      ? reasoningTurns[0]
      : reasoningTurns.map((t, i) => `[turn ${i + 1}]\n${t}`).join("\n\n");

  return {
    output: lastResult?.output ?? "",
    toolCalls: aggregateCalls,
    reasoningText: concatenatedReasoning,
    latencyMs: totalLatency,
    inputTokens: totalInput,
    outputTokens: totalOutput,
    costUsd: totalCost,
    provider: lastResult?.provider ?? provider.name,
    model: lastResult?.model ?? provider.model,
    error: lastResult?.error,
    turns,
    truncated,
  };
}

function mergeParams(
  ...layers: (Record<string, unknown> | undefined)[]
): Record<string, unknown> | undefined {
  const merged: Record<string, unknown> = {};
  let any = false;
  for (const layer of layers) {
    if (!layer) continue;
    Object.assign(merged, layer);
    any = true;
  }
  return any ? merged : undefined;
}

function timingFrom(result: ProviderResult): { total_tokens: number; duration_ms: number } {
  return {
    total_tokens: (result.inputTokens ?? 0) + (result.outputTokens ?? 0),
    duration_ms: result.latencyMs ?? 0,
  };
}

export async function runEval(args: RunEvalArgs): Promise<RunEvalResult> {
  if (args.modes.length === 0) throw new Error("runEval requires at least one mode");

  const slug = evalSlug(args.eval, args.index);
  const evalDir = path.join(args.evalRootDir ?? path.join(args.workspace, `iteration-${args.iteration}`), slug);
  const result: RunEvalResult = { slug, modes: {} as RunEvalResult["modes"] };
  const evalIndex = args.index ?? 0;

  // Resolve effective tools / tool_choice / params for this case once.
  // Precedence (low → high): caller programmatic args, skill defaults, eval-level.
  const effectiveTools: ToolDef[] | undefined =
    args.eval.tools ?? args.skill.defaults?.tools;
  const effectiveToolChoice: ToolChoice | undefined =
    args.eval.tool_choice ?? (effectiveTools && effectiveTools.length > 0 ? "auto" : undefined);
  const effectiveTargetParams = mergeParams(
    args.targetParams,
    args.skill.defaults?.target?.params,
    args.eval.params
  );
  const effectiveJudgeParams = mergeParams(
    args.judgeParams,
    args.skill.defaults?.judge?.params
  );

  for (const mode of args.modes) {
    const runDir = path.join(evalDir, mode);
    const outputDir = path.join(runDir, "outputs");
    const evalFiles = mode === "with_skill" ? readEvalFiles(args.skill, args.eval) : [];
    const system = mode === "with_skill" ? renderSkillSystemMessage(args.skill) : undefined;
    const userMessage = args.eval.prompt;

    args.onEvent?.({
      type: "eval-start",
      skill: args.skill.name,
      evalIndex,
      evalSlug: slug,
      evalName: args.eval.name,
      evalId: args.eval.id,
      mode,
      system,
      user: userMessage,
      fileCount: evalFiles.length,
      tools: effectiveTools,
      toolChoice: effectiveToolChoice,
    });

    const completion = await completeWithToolLoop({
      provider: args.target.provider,
      system,
      user: userMessage,
      attachments: evalFiles,
      tools: effectiveTools,
      toolChoice: effectiveToolChoice,
      params: effectiveTargetParams,
      maxTurns: args.maxToolTurns ?? 6,
      toolMocks: args.skill.defaults?.tool_mocks,
    });
    const rawOutput = completion.error ? `ERROR: ${completion.error}` : completion.output;
    const toolCalls = completion.toolCalls.length > 0 ? completion.toolCalls : undefined;
    const assertions =
      args.eval.assertions && args.eval.assertions.length > 0
        ? args.eval.assertions
        : args.eval.expected_output
          ? [`The output satisfies this expected output: ${args.eval.expected_output}`]
          : [];
    const { grading, judgePrompt } = await gradeOutputs({
      modelOutput: rawOutput,
      reasoningText: completion.reasoningText,
      assertions,
      toolCalls,
      toolAssertions: args.eval.tool_assertions,
      judge: args.judge,
      judgeParams: effectiveJudgeParams,
      gradingPrompt: args.gradingPrompt,
    });
    const timing = timingFrom(completion);
    writeRunArtifacts(
      runDir,
      timing,
      grading,
      rawOutput,
      [{ path: "output.txt", content: rawOutput }],
      {
        system,
        user: userMessage,
        judgePrompt,
        fileCount: evalFiles.length,
        tools: effectiveTools,
        tool_choice: effectiveToolChoice,
      },
      toolCalls
    );

    result.modes[mode] = {
      outputDir,
      timing,
      grading,
      rawOutput,
      toolCalls,
      system,
      user: userMessage,
      fileCount: evalFiles.length,
      judgePrompt,
      tools: effectiveTools,
      toolChoice: effectiveToolChoice,
    };

    args.onEvent?.({
      type: "eval-end",
      skill: args.skill.name,
      evalIndex,
      evalSlug: slug,
      evalName: args.eval.name,
      evalId: args.eval.id,
      mode,
      output: rawOutput,
      timing,
      grading,
      judgePrompt,
      toolCalls,
    });
  }

  return result;
}
