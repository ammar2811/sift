/**
 * Contrast, checked against the tokens the stylesheet actually ships.
 *
 * axe cannot do this in the test suite: jsdom performs no layout and computes no colours,
 * so its colour-contrast rule opts out rather than passing. Checking the pairs directly
 * is better than nothing and, in one respect, better than a browser audit - a rendered
 * page only exercises the combinations that happen to be on screen, while this covers
 * every pair the design intends, in both themes, including states a screenshot misses.
 *
 * The ratios come from WCAG 2.1: 4.5:1 for body text, 3:1 for large text and for the
 * boundary of a user-interface component.
 *
 * The tokens are parsed out of styles.css rather than restated here, so a colour that
 * changes without its contrast being reconsidered fails this test rather than passing a
 * copy of itself.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

// Read from disk rather than imported. Vitest stubs CSS modules, so `?raw` yields an
// empty string here, and `import.meta.url` is not a file: URL under jsdom. The path is
// relative to the vitest root, which is apps/web; the parser guard at the bottom of this
// file fails loudly if that ever stops resolving.
const CSS = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");

/** Pull `--name: #value;` pairs out of a block of CSS. */
function tokensIn(block: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const match of block.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{3,8})\s*;/g)) {
    const name = match[1];
    const value = match[2];
    if (name && value) out[name] = value;
  }
  return out;
}

/** The `:root` block, and the dark overrides that follow `prefers-color-scheme: dark`. */
function themes(): { light: Record<string, string>; dark: Record<string, string> } {
  const rootStart = CSS.indexOf(":root {");
  const light = tokensIn(CSS.slice(rootStart, CSS.indexOf("}", rootStart)));

  const darkAt = CSS.indexOf("@media (prefers-color-scheme: dark)");
  const darkBlock = CSS.slice(darkAt, CSS.indexOf("\n}", CSS.indexOf(":root", darkAt)));
  return { light, dark: { ...light, ...tokensIn(darkBlock) } };
}

function channel(hex: string): [number, number, number] {
  const clean = hex.replace("#", "");
  const full =
    clean.length === 3
      ? clean
          .split("")
          .map((c) => c + c)
          .join("")
      : clean;
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16) / 255) as [number, number, number];
}

/** WCAG relative luminance. */
function luminance(hex: string): number {
  const [r, g, b] = channel(hex).map((v) =>
    v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4,
  ) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
}

const { light, dark } = themes();

/** [foreground, background, minimum ratio, what it is] */
const PAIRS: [string, string, number, string][] = [
  ["text", "bg", 4.5, "body text on the page"],
  ["text", "surface", 4.5, "body text on a card"],
  ["text", "surface-sunken", 4.5, "preformatted text on its panel"],
  ["text-muted", "bg", 4.5, "secondary text on the page"],
  ["text-muted", "surface", 4.5, "secondary text on a card"],
  ["text-faint", "bg", 4.5, "the trajectory and usage lines"],
  ["text-faint", "surface", 4.5, "faint text on a card"],
  ["accent", "bg", 4.5, "links on the page"],
  ["accent", "surface", 4.5, "citation links on a card"],
  ["accent", "accent-surface", 4.5, "a source chip"],
  ["warn-text", "warn-surface", 4.5, "the supersession callout"],
  // Component boundaries and focus indication: 3:1 under WCAG 2.1 (1.4.11).
  ["border-strong", "bg", 3, "an input border against the page"],
  ["focus", "bg", 3, "the focus ring against the page"],
  ["focus", "surface", 3, "the focus ring against a card"],
];

describe.each([
  ["light", light],
  ["dark", dark],
])("%s theme contrast", (_name, tokens) => {
  it.each(PAIRS)("%s on %s is at least %s:1 (%s)", (fg, bg, min) => {
    expect(tokens[fg], `--${fg} is not defined`).toBeDefined();
    expect(tokens[bg], `--${bg} is not defined`).toBeDefined();
    expect(contrast(tokens[fg]!, tokens[bg]!)).toBeGreaterThanOrEqual(min);
  });
});

describe("the token set itself", () => {
  it("defines a dark value for every colour the light theme has", () => {
    // A token defined only in :root silently keeps its light value in dark mode, which
    // is how a single unreadable element survives a theme switch unnoticed.
    for (const name of Object.keys(light)) {
      expect(dark[name], `--${name} has no dark-theme value`).toBeDefined();
    }
  });

  it("parsed both themes at all", () => {
    // Guards the parser above: if styles.css is restructured and these come back empty,
    // every assertion would vacuously pass.
    expect(Object.keys(light).length).toBeGreaterThan(10);
    expect(Object.keys(dark).length).toBeGreaterThan(10);
    expect(dark["bg"]).not.toBe(light["bg"]);
  });
});
