# Accessibility

The project claims WCAG 2.1 AA. This is what that claim rests on, what an audit found,
and what is still untested.

Every check below runs in CI. That is the point of the arrangement: a check that runs on
every commit holds, and a check someone remembers to run before a demo is a check that
happened once. `pnpm lint`, `pnpm test` and `pnpm typecheck` all gate the build.

## What enforces it

| | |
|---|---|
| `eslint-plugin-jsx-a11y` at **strict**, `--max-warnings 0` | Static rules over the JSX. Catches a missing label, a click handler on a div, an `aria-*` that is not valid for its role. |
| **axe-core**, `src/a11y.test.tsx` | The rendered tree, in five states: landing, streaming, answered, search results with a superseded hit, and the error state. WCAG 2.0/2.1 A and AA only. |
| **Contrast**, `src/contrast.test.ts` | 15 foreground/background pairs in both themes, parsed out of `styles.css` so a colour cannot change without its contrast being rechecked. |
| **Behaviour**, `src/App.test.tsx` | Skip link is the first tab stop, `/` focuses search, `/` does not hijack while typing, every filter has an accessible name. |

Two of those deserve a note on why they are shaped as they are.

jsx-a11y sees JSX in isolation. It cannot tell whether the *rendered* tree has a sane
heading order, whether a live region ends up nested inside another, or whether an element
that gets its role at runtime also gets the attributes that role requires. axe sees the
tree the browser would build, which is the second class of problem.

axe runs under jsdom, which performs no layout and computes no colour, so its
`color-contrast` rule cannot evaluate and is explicitly disabled rather than silently
skipped. Contrast is therefore checked separately against the tokens the stylesheet
ships. That is worse than a browser audit in one way and better in another: it cannot see
what is actually painted, but it covers every pair the design intends rather than only the
combinations that happen to be on screen when someone runs the audit.

Both suites carry a guard against becoming vacuous. The axe test asserts it evaluated
more than ten rules, so a misconfigured `runOnly` cannot report zero violations forever.
The contrast test asserts both themes parsed and that they differ, so a restructured
stylesheet cannot make every assertion pass over an empty token set.

## What the audit found

axe reported **no violations** in any of the five states. That result is only worth
stating because the same configuration, pointed at `<img src="x.png"><input type="text">`,
reports `image-alt` and `label` - the audit was checked against a known-bad fragment
before its clean result was believed.

The contrast audit found **four real failures**, all of which shipped:

| token pair | where it shows | before | after | required |
|---|---|---|---|---|
| `--text-faint` on `--bg` (light) | tool trajectory, usage row, corpus badge | 3.59:1 | **4.89:1** | 4.5:1 |
| `--text-faint` on `--surface` (light) | faint text inside a card | 3.72:1 | **5.06:1** | 4.5:1 |
| `--border-strong` on `--bg` (light) | the search input's own border | 1.61:1 | **3.31:1** | 3:1 |
| `--text-faint` on `--surface` (dark) | same, dark theme | 4.47:1 | **4.97:1** | 4.5:1 |

The first two are the lines that report what the agent did and what the answer cost -
small mono text, exactly the kind that gets set too light. The third is WCAG 2.1's
1.4.11: the boundary of a user-interface control needs 3:1 against what is behind it, and
a search box whose border is 1.6:1 is a box a low-vision user cannot find. `--border-strong`
in dark mode was fixed at the same time, from 1.91:1 to 3.19:1, though nothing failed on it
because no test had asserted it before.

None of these would have been caught by jsx-a11y, and none were visible to anyone with
ordinary vision looking at the page. They were found by computing them.

## Decisions worth recording

**The streaming answer is deliberately not a live region.** `aria-live` on text that
arrives token by token announces the answer a word at a time, which is unusable. The
answer element carries `aria-busy` while streaming, and a single `role="status"` region
announces once when the answer settles: *"Answer complete, 1 source cited, in 5.4
seconds."* There is exactly one status region on the page, asserted by test, so two
announcements cannot race.

**The segmented controls are real radios.** Mode and retrieval mode look like button
groups and are `<input type="radio">` underneath, styled with `:has(input:checked)`. Arrow
keys work, the group has an accessible name via `useId` and `aria-labelledby`, and the
state is exposed without any ARIA of its own.

**Focus is never suppressed.** One `:focus-visible` rule, applied globally, with an
outline offset so it is visible against every surface. The focus colour is checked at 3:1
against both the page and a card.

**The skip link is the first tab stop** and targets `#results`, which carries
`tabIndex={-1}` so it can receive focus. Tab order was verified in a browser: skip link,
search box, submit, mode radios, then the example queries. No positive `tabindex`
anywhere.

**Motion respects the setting.** The streaming caret and the loading shimmer both stop
under `prefers-reduced-motion: reduce`, and the caret is held visible rather than hidden,
so it still marks where text is arriving.

**External links say so.** Every link to rfc-editor.org appends a visually hidden
"(opens on rfc-editor.org in a new tab)" and carries `rel="noreferrer"`.

## What has not been done

The honest gap. **No manual screen-reader pass has been run** - not NVDA, not VoiceOver,
not Orca. Everything above is automated, and automated accessibility testing is generally
reckoned to catch something like a third to a half of real barriers. Roles and names are
asserted; whether the result is *pleasant* to listen to is not, and the announcement
copy in particular has never been heard by anyone.

Also missing:

- No testing at 200% zoom or 320px width. The layout uses relative units and should
  reflow, but "should" is not a measurement.
- No Windows High Contrast Mode check.
- The `color-contrast` rule is disabled under jsdom, so nothing verifies that the
  *painted* colours match the tokens - only that the tokens themselves are sound.
- Contrast is checked for the pairs listed in the test. A pair nobody thought to list is
  a pair nobody checked.
