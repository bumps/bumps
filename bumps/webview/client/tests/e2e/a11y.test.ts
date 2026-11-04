import AxeBuilder from "@axe-core/playwright";
import { expect, panes, test } from "./fixtures";

/**
 * Known serious violations, not yet fixed. Remove an entry once it's fixed so
 * the rule is enforced again. Moderate/minor rules (landmarks, an <h1>, empty
 * table headers) aren't enforced yet.
 */
const KNOWN: Record<string, string> = {
  // TODO: vue-json-pretty's default string colour fails contrast (Model panel)
  "color-contrast": "vue-json-pretty theme",
  // TODO: panel content area scrolls but isn't keyboard focusable (Model panel)
  "scrollable-region-focusable": "panel .tab-content needs tabindex or focusable content",
};

for (const title of ["Data", "Summary", "Log", "History", "Convergence", "Model", "Parameters"]) {
  test(`${title} panel has no serious a11y violations`, async ({ page }) => {
    const tab = panes(page).left.getByRole("tab", { name: title, exact: true });
    await tab.click();
    await expect(tab).toHaveClass(/active/);
    await page.waitForTimeout(500); // let the panel fetch and draw

    const { violations } = await new AxeBuilder({ page }).analyze();
    const serious = violations
      .filter((v) => (v.impact === "serious" || v.impact === "critical") && !(v.id in KNOWN))
      .map((v) => `${v.id} (${v.impact}): ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
    expect(serious).toEqual([]);
  });
}
